#!/usr/bin/env python3
import os
from aws_cdk import App, Environment

from stacks.banking_rag_stack import BankingRagStack

app = App()

account = os.getenv("CDK_DEFAULT_ACCOUNT")
region = os.getenv("CDK_DEFAULT_REGION")

BankingRagStack(
    app,
    "BankingRagStack",
    env=Environment(account=account, region=region),
)

app.synth()
