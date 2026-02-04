# Operations Runbook (MVP)

This service runs on AWS ECS Fargate behind an Application Load Balancer (ALB), provisioned with CDK.

## 1) Quick health checks

- Liveness: `GET /health` should return `200`.
- Readiness: `GET /ready` should return `200` with manifest details.

```bash
curl -i http://<alb-dns>/health
curl -i http://<alb-dns>/ready
```

## 2) Logs (CloudWatch)

### Console

- Open CloudWatch Logs and search for log group `/ecs/banking-rag`.
- Open the latest stream with prefix `banking-rag`.
- Filter common patterns: `status=500`, `rate_limited`, or a specific `request_id`.

### CLI

```bash
aws logs tail /ecs/banking-rag --since 30m --follow --region <aws-region>
```

Optional: list streams first.

```bash
aws logs describe-log-streams \
  --log-group-name /ecs/banking-rag \
  --order-by LastEventTime \
  --descending \
  --region <aws-region>
```

## 3) Common failure modes

### `/ready` returns 503

Typical causes:
- `artifacts/vector_db/` missing or empty
- `artifacts/manifest.json` missing or invalid
- Embedding provider/model in manifest does not match runtime config

What to do:
1. Check `/ecs/banking-rag` logs for readiness detail.
2. Rebuild artifacts (`make build-index`) and ensure `artifacts/` is in the image.
3. Redeploy and re-check `/ready`.

### `/ask` returns 401 or 403

- `401`: `X-API-Key` header missing.
- `403`: `X-API-Key` value does not match `DEMO_API_KEY`.

What to do:
1. Confirm caller sends `X-API-Key`.
2. Confirm task env var `DEMO_API_KEY` is set to the expected value.
3. Redeploy task if key/config changed.

## 4) Redeploy and image tagging (ECR)

Use two tags for each release:
- Immutable: `git-<short_sha>` (traceable)
- Mutable: `latest` (current deploy pointer)

```bash
export AWS_REGION=<aws-region>
export AWS_ACCOUNT_ID=<account-id>
export ECR_REPO=banking-rag
export GIT_SHA=$(git rev-parse --short HEAD)
export ECR_URI=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}

aws ecr get-login-password --region ${AWS_REGION} \
  | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

docker build -t ${ECR_REPO}:${GIT_SHA} .
docker tag ${ECR_REPO}:${GIT_SHA} ${ECR_URI}:git-${GIT_SHA}
docker tag ${ECR_REPO}:${GIT_SHA} ${ECR_URI}:latest

docker push ${ECR_URI}:git-${GIT_SHA}
docker push ${ECR_URI}:latest
```

Force a new ECS deployment:

```bash
aws ecs update-service \
  --cluster <ecs-cluster-name-or-arn> \
  --service <ecs-service-name-or-arn> \
  --force-new-deployment \
  --region ${AWS_REGION}
```

Verify rollout:

```bash
curl -i http://<alb-dns>/health
curl -i http://<alb-dns>/ready
```

## 5) Metrics that matter (minimum)

Create CloudWatch alarms for:

1. ALB `HTTPCode_Target_5XX_Count`
   - Alert when > 0 for 5 minutes.
2. ALB `TargetResponseTime` (prefer p95)
   - Alert on sustained latency above baseline.
3. ECS `CPUUtilization`
   - Alert when > 80% for 10 minutes.
4. ECS `MemoryUtilization`
   - Alert when > 85% for 10 minutes.

## 6) Definition of done

A teammate can operate this service without guessing by using this doc alone:

1. Find logs and identify failed requests.
2. Diagnose `/ready` 503 and `/ask` 401/403.
3. Build, tag, and push a traceable image.
4. Trigger ECS redeploy and verify `/health` + `/ready`.
5. Check ALB/ECS metrics and know when to alert.
