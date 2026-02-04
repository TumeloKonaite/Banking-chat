import os

from aws_cdk import (
    Stack,
    CfnOutput,
    Duration,
    RemovalPolicy,
    aws_ec2 as ec2,
    aws_ecr as ecr,
    aws_ecs as ecs,
    aws_logs as logs,
    aws_secretsmanager as secretsmanager,
)
from aws_cdk.aws_ecs_patterns import (
    ApplicationLoadBalancedFargateService,
    ApplicationLoadBalancedTaskImageOptions,
)
from constructs import Construct


class BankingRagStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        demo_api_key = os.getenv("DEMO_API_KEY")
        if not demo_api_key:
            raise ValueError("DEMO_API_KEY must be set for the task to start.")
        openai_secret_name = os.getenv("OPENAI_SECRET_NAME", "banking-rag/openai-api-key")
        openai_api_secret = secretsmanager.Secret.from_secret_name_v2(
            self,
            "OpenAiApiKeySecret",
            openai_secret_name,
        )

        # VPC (MVP: 2 AZs)
        vpc = ec2.Vpc(
            self,
            "BankingRagVpc",
            max_azs=2,
            nat_gateways=1,
        )

        # ECS cluster
        cluster = ecs.Cluster(self, "BankingRagCluster", vpc=vpc)

        # ECR repo (you'll push your Docker image here)
        repo = ecr.Repository(
            self,
            "BankingRagRepo",
            repository_name="banking-rag",
            removal_policy=RemovalPolicy.DESTROY,  # MVP only; change to RETAIN later
            empty_on_delete=True,                  # MVP only
        )

        # CloudWatch logs
        log_group = logs.LogGroup(
            self,
            "BankingRagLogGroup",
            log_group_name="/ecs/banking-rag",
            retention=logs.RetentionDays.ONE_WEEK,
            removal_policy=RemovalPolicy.DESTROY,  # MVP only
        )

        # Fargate service behind an ALB
        # IMPORTANT: set container_port to whatever your app listens on (e.g., 8000)
        container_port = 8000

        service = ApplicationLoadBalancedFargateService(
            self,
            "BankingRagService",
            cluster=cluster,
            cpu=512,
            memory_limit_mib=1024,
            desired_count=1,
            public_load_balancer=True,
            task_image_options=ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_ecr_repository(repo, tag="latest"),
                container_port=container_port,
                enable_logging=True,
                log_driver=ecs.LogDrivers.aws_logs(
                    stream_prefix="banking-rag",
                    log_group=log_group,
                ),
                environment={
                    "DEMO_MODE": "true",
                    "DEMO_API_KEY": demo_api_key,
                    # Add non-secret env vars here
                },
                secrets={
                    "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(openai_api_secret),
                },
            ),
            health_check_grace_period=Duration.seconds(60),
        )

        # Health check path (your app must return 200 quickly)
        service.target_group.configure_health_check(
            path="/health",
            healthy_http_codes="200",
            interval=Duration.seconds(30),
        )

        # Outputs
        CfnOutput(self, "AlbDnsName", value=service.load_balancer.load_balancer_dns_name)
        CfnOutput(self, "EcrRepoUri", value=repo.repository_uri)
        CfnOutput(self, "GradioUrl", value=f"http://{service.load_balancer.load_balancer_dns_name}")
        CfnOutput(self, "DocsUrl", value=f"http://{service.load_balancer.load_balancer_dns_name}/api/docs")
        CfnOutput(self, "HealthUrl", value=f"http://{service.load_balancer.load_balancer_dns_name}/health")
