# -*- coding: utf-8 -*-
# @Time    : 9/24/20 1:51 PM
# @Author  : Jackie
# @File    : create_endpoint.py
# @Software: PyCharm

import boto3
import argparse
from sagemaker import get_execution_role

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(
        description="create SageMaker endpoint"
    )
    parser.add_argument(
        "-e",
        "--endpoint_ecr_image_path",
        type=str,
        help=" ",
        default="064542430558.dkr.ecr.us-east-2.amazonaws.com/pegasus:latest"
    )
    parser.add_argument(
        "-en",
        "--endpoint_name",
        type=str,
        nargs="?",
        help="When set, this argument ",
        default="pegasus"
    )
    parser.add_argument(
        "-i",
        "--instance_type",
        type=str,
        nargs="?",
        help="When set, this argument ",
        default="ml.g4dn.xlarge"
    )

    return parser.parse_args()

def is_endpoint_running(endpoint_name):
    """
    Content of check_name could be "InService" or other.
    if the named endpoint doesn't exist then return None.
    """
    client = boto3.client('sagemaker')
    endpoints = client.list_endpoints()
    endpoint_name_list = [(ep["EndpointName"], ep["EndpointStatus"]) for ep in endpoints["Endpoints"]]
    for check_name in endpoint_name_list:
        if endpoint_name == check_name[0]:
            return check_name[1]
    return None

def deploy_endpoint():
    args = parse_arguments()
    if is_endpoint_running(args.endpoint_name) is not None:
        print("Endpoint already exist and will return.")
        return

    try:
        role = get_execution_role()
    except Exception as e:
        print("SageMaker Role doesn't exist.")

    try:
        sm = boto3.Session().client('sagemaker')
        primary_container = {'Image': args.endpoint_ecr_image_path}
        print("model_name: ", args.endpoint_name)
        print("endpoint_ecr_image_path: ", args.endpoint_ecr_image_path)
        create_model_response = sm.create_model(ModelName=args.endpoint_name,
                                                ExecutionRoleArn=role,
                                                PrimaryContainer=primary_container)

        # create endpoint config
        endpoint_config_name = args.endpoint_name + '-config'
        create_endpoint_config_response = sm.create_endpoint_config(EndpointConfigName=endpoint_config_name,
                                                                    ProductionVariants=[{
                                                                        'InstanceType': args.instance_type,
                                                                        'InitialVariantWeight': 1,
                                                                        'InitialInstanceCount': 1,
                                                                        'ModelName': args.endpoint_name,
                                                                        'VariantName': 'AllTraffic'}])

        # create endpoint
        create_endpoint_response = sm.create_endpoint(
            EndpointName=args.endpoint_name,
            EndpointConfigName=endpoint_config_name)

    except Exception as e:
        print("!!! Cannot create endpoint - Exception is >> {}".format(e))
        if type(e).__name__ == "StateMachineAlreadyExists":
            print("Skip sf creation because it is created before.")
        else:
            raise e

    print("<<< Completed model endpoint deployment. " + str(args.endpoint_name))

if __name__ == '__main__':
    deploy_endpoint()