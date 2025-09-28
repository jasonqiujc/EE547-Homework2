import os
import sys
import json
import argparse
from datetime import datetime

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError



def parse_args():
    ap = argparse.ArgumentParser(description="List IAM/EC2/S3/SecurityGroup resources")
    ap.add_argument("--region", help="AWS region, e.g. us-east-1")
    ap.add_argument("--output", help="Path to write result (otherwise print to stdout)")
    ap.add_argument("--format", choices=["json", "table"], default="json", help="Output format")
    return ap.parse_args()


def error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)


def warn(msg):
    print(f"[WARNING] {msg}", file=sys.stderr)


def validate_region(session, region):
    """Check region is valid for EC2 (simple universal list)."""
    if not region:
        return session.region_name or "us-east-1"
    regions = session.get_available_regions("ec2")
    if region not in regions:
        error(f"Invalid region: {region}")
        sys.exit(1)
    return region


def get_account_info(session, region):
    """Use STS to confirm credentials and fetch account id / caller ARN."""
    sts = session.client("sts", region_name=region)
    ident = sts.get_caller_identity()
    return {
        "account_id": ident["Account"],
        "user_arn": ident["Arn"],
        "region": region,
        "scan_timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


def safe_call(fn, context):
    """Call AWS API and convert common errors to warnings (continue)."""
    try:
        return fn()
    except EndpointConnectionError:
        warn(f"Network timeout while calling {context}")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        msg = e.response.get("Error", {}).get("Message", str(e))
        # Access denied -> warn and continue; other errors -> warn as well
        level = "WARNING" if code in ("AccessDenied", "UnauthorizedOperation") else "WARNING"
        print(f"[{level}] {msg} ({context})", file=sys.stderr)
    return None



def list_iam_users(session, region):
    iam = session.client("iam", region_name=region)
    users = []
    resp = safe_call(lambda: iam.list_users(), "iam:list_users")
    if not resp:
        return users
    for u in resp.get("Users", []):
        item = {
            "username": u["UserName"],
            "user_id": u["UserId"],
            "arn": u["Arn"],
            "create_date": u["CreateDate"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_activity": None,
            "attached_policies": [],
        }
        gu = safe_call(lambda: iam.get_user(UserName=u["UserName"]), f"iam:get_user {u['UserName']}")
        if gu and gu.get("User", {}).get("PasswordLastUsed"):
            item["last_activity"] = gu["User"]["PasswordLastUsed"].strftime("%Y-%m-%dT%H:%M:%SZ")
        ap = safe_call(
            lambda: iam.list_attached_user_policies(UserName=u["UserName"]),
            f"iam:list_attached_user_policies {u['UserName']}"
        )
        if ap:
            item["attached_policies"] = [
                {"policy_name": p["PolicyName"], "policy_arn": p["PolicyArn"]}
                for p in ap.get("AttachedPolicies", [])
            ]
        users.append(item)
    return users


def list_ec2_instances(session, region):
    ec2 = session.client("ec2", region_name=region)
    instances = []
    resp = safe_call(lambda: ec2.describe_instances(), "ec2:describe_instances")
    if not resp:
        return instances

    ami_name_cache = {}

    for r in resp.get("Reservations", []):
        for inst in r.get("Instances", []):
            ami_id = inst.get("ImageId")
            ami_name = None
            if ami_id and ami_id not in ami_name_cache:
                img = safe_call(lambda: ec2.describe_images(ImageIds=[ami_id]), f"ec2:describe_images {ami_id}")
                if img and img.get("Images"):
                    ami_name_cache[ami_id] = img["Images"][0].get("Name")
                else:
                    ami_name_cache[ami_id] = None
            ami_name = ami_name_cache.get(ami_id)

            item = {
                "instance_id": inst["InstanceId"],
                "instance_type": inst["InstanceType"],
                "state": inst["State"]["Name"],
                "public_ip": inst.get("PublicIpAddress"),
                "private_ip": inst.get("PrivateIpAddress"),
                "availability_zone": inst["Placement"]["AvailabilityZone"],
                "launch_time": inst["LaunchTime"].strftime("%Y-%m-%dT%H:%M:%SZ"),
                "ami_id": ami_id,
                "ami_name": ami_name,
                "security_groups": [g["GroupId"] for g in inst.get("SecurityGroups", [])],
                "tags": {t["Key"]: t["Value"] for t in inst.get("Tags", [])} if inst.get("Tags") else {},
            }
            instances.append(item)
    return instances


def list_s3_buckets(session, region):
    s3 = session.client("s3", region_name=region)
    out = []
    resp = safe_call(lambda: s3.list_buckets(), "s3:list_buckets")
    if not resp:
        return out
    for b in resp.get("Buckets", []):
        name = b["Name"]
        loc = safe_call(lambda: s3.get_bucket_location(Bucket=name), f"s3:get_bucket_location {name}")
        bucket_region = (loc or {}).get("LocationConstraint") or "us-east-1"
        total_bytes, total_objects = 0, 0
        try:
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=name):
                for obj in page.get("Contents", []) or []:
                    total_objects += 1
                    total_bytes += int(obj.get("Size", 0))
        except ClientError as e:
            warn(f"Cannot list objects for bucket '{name}': {e.response['Error'].get('Message','')}")
        out.append({
            "bucket_name": name,
            "creation_date": b["CreationDate"].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "region": bucket_region,
            "object_count": total_objects,
            "size_bytes": total_bytes
        })
    return out


def list_security_groups(session, region):
    ec2 = session.client("ec2", region_name=region)
    resp = safe_call(lambda: ec2.describe_security_groups(), "ec2:describe_security_groups")
    if not resp:
        return []

    def rules_to_list(perms, inbound=True):
        rules = []
        for p in perms:
            proto = p.get("IpProtocol", "all")
            from_p, to_p = p.get("FromPort"), p.get("ToPort")
            port_range = "all" if from_p is None else f"{from_p}-{to_p}"
            cidrs = [x.get("CidrIp") for x in p.get("IpRanges", [])]
            key = "source" if inbound else "destination"
            for cidr in cidrs:
                rules.append({"protocol": proto, "port_range": port_range, key: cidr})
        return rules

    out = []
    for g in resp.get("SecurityGroups", []):
        out.append({
            "group_id": g["GroupId"],
            "group_name": g.get("GroupName"),
            "description": g.get("Description"),
            "vpc_id": g.get("VpcId"),
            "inbound_rules": rules_to_list(g.get("IpPermissions", []), inbound=True),
            "outbound_rules": rules_to_list(g.get("IpPermissionsEgress", []), inbound=False),
        })
    return out



def to_json_blob(account, users, instances, buckets, sgs):
    return {
        "account_info": account,
        "resources": {
            "iam_users": users,
            "ec2_instances": instances,
            "s3_buckets": buckets,
            "security_groups": sgs
        },
        "summary": {
            "total_users": len(users),
            "instances_total": len(instances),
            "instances_running": sum(1 for i in instances if i["state"] == "running"),
            "total_buckets": len(buckets),
            "security_groups": len(sgs),
        }
    }


def to_table_text(account, users, instances, buckets, sgs):
    lines = []
    lines.append(f"AWS Account: {account['account_id']} | Region: {account['region']}")
    lines.append(f"Scan Time  : {account['scan_timestamp']}\n")

    lines.append(f"IAM USERS ({len(users)})")
    lines.append("Username            Created              LastActivity         Policies")
    for u in users:
        lines.append(f"{u['username']:<20}{u['create_date']:<20}{(u['last_activity'] or '-'):<20}{len(u['attached_policies'])}")
    lines.append("")

    running = sum(1 for i in instances if i['state'] == 'running')
    lines.append(f"EC2 INSTANCES (total {len(instances)}, running {running})")
    lines.append("InstanceId           Type        State      PublicIP         LaunchTime")
    for i in instances:
        lines.append(f"{i['instance_id']:<20}{i['instance_type']:<11}{i['state']:<10}{(i['public_ip'] or '-'):<16}{i['launch_time']}")
    lines.append("")

    lines.append(f"S3 BUCKETS ({len(buckets)})")
    lines.append("BucketName                 Region        Created        Objects   Size(MB)")
    for b in buckets:
        size_mb = round(b['size_bytes'] / 1e6, 1)
        lines.append(f"{b['bucket_name']:<26}{b['region']:<13}{b['creation_date'][:10]:<14}{b['object_count']:<9}{size_mb:<7}")
    lines.append("")

    lines.append(f"SECURITY GROUPS ({len(sgs)})")
    lines.append("GroupId             Name             VPC              InboundRules")
    for g in sgs:
        lines.append(f"{g['group_id']:<20}{(g['group_name'] or '-'):<16}{(g['vpc_id'] or '-'):<17}{len(g['inbound_rules'])}")
    return "\n".join(lines)



def main():
    args = parse_args()
    session = boto3.session.Session()

    region = validate_region(session, args.region)

    try:
        account = get_account_info(session, region)
    except ClientError as e:
        error(e.response.get("Error", {}).get("Message", str(e)))
        sys.exit(1)

    users = list_iam_users(session, region) or []
    instances = list_ec2_instances(session, region) or []
    buckets = list_s3_buckets(session, region) or []
    sgs = list_security_groups(session, region) or []

    if args.format == "json":
        payload = json.dumps(to_json_blob(account, users, instances, buckets, sgs), indent=2)
    else:
        payload = to_table_text(account, users, instances, buckets, sgs)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
