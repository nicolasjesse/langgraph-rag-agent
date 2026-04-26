#!/usr/bin/env bash
# Deploy the LangGraph RAG agent to AWS Lambda + API Gateway.
#
# Idempotent: safe to re-run. Updates the image on subsequent runs.
# Outputs LAMBDA_API_URL and LAMBDA_API_KEY into .env (gitignored).
#
# Requires: aws-cli configured, docker, .env populated with
#   ANTHROPIC_API_KEY, OPENAI_API_KEY.

set -euo pipefail

# --- Config ------------------------------------------------------------------
REGION="${AWS_REGION:-us-east-1}"
FN="langgraph-rag-agent"
ROLE_NAME="${FN}-lambda-role"
PLAN_NAME="${FN}-plan"
KEY_NAME="${FN}-key"

# Daily quota + throttle — keep these tight. Worst-case spend if the
# key leaks is roughly: DAILY_QUOTA * cost-per-call (~$0.05).
DAILY_QUOTA=30
RATE_LIMIT=1
BURST_LIMIT=2

# --- Pre-flight --------------------------------------------------------------
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${FN}"
ROLE_ARN="arn:aws:iam::${ACCOUNT}:role/${ROLE_NAME}"
LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT}:function:${FN}"

if [[ ! -f .env ]]; then
  echo "ERROR: .env file not found. Copy .env.example and fill in keys." >&2
  exit 1
fi
ANTHROPIC=$(grep '^ANTHROPIC_API_KEY=' .env | head -1 | cut -d= -f2-)
OPENAI=$(grep '^OPENAI_API_KEY=' .env | head -1 | cut -d= -f2-)
if [[ -z "${ANTHROPIC}" || -z "${OPENAI}" ]]; then
  echo "ERROR: ANTHROPIC_API_KEY or OPENAI_API_KEY missing from .env" >&2
  exit 1
fi

# --- 1. Build + push image ---------------------------------------------------
echo ">> Building image (Lambda-compatible manifest)..."
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -t "${FN}:latest" --load .

echo ">> Ensuring ECR repo exists..."
aws ecr describe-repositories --repository-names "${FN}" --region "${REGION}" >/dev/null 2>&1 \
  || aws ecr create-repository --repository-name "${FN}" --region "${REGION}" \
       --image-scanning-configuration scanOnPush=true >/dev/null

echo ">> Logging Docker into ECR..."
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

echo ">> Pushing image..."
docker tag "${FN}:latest" "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"

# --- 2. Lambda execution role -----------------------------------------------
echo ">> Ensuring Lambda execution role..."
TRUST='{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1 \
  || aws iam create-role --role-name "${ROLE_NAME}" --assume-role-policy-document "${TRUST}" >/dev/null
aws iam attach-role-policy --role-name "${ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null
sleep 5  # IAM propagation

# --- 3. Lambda function ------------------------------------------------------
echo ">> Creating or updating Lambda function..."
if aws lambda get-function --function-name "${FN}" --region "${REGION}" >/dev/null 2>&1; then
  aws lambda update-function-code --function-name "${FN}" \
    --image-uri "${ECR_URI}:latest" --region "${REGION}" >/dev/null
  aws lambda wait function-updated --function-name "${FN}" --region "${REGION}"
  aws lambda update-function-configuration --function-name "${FN}" --region "${REGION}" \
    --timeout 60 --memory-size 1024 \
    --environment "Variables={ANTHROPIC_API_KEY=${ANTHROPIC},OPENAI_API_KEY=${OPENAI}}" >/dev/null
else
  aws lambda create-function --function-name "${FN}" --region "${REGION}" \
    --package-type Image --code "ImageUri=${ECR_URI}:latest" \
    --role "${ROLE_ARN}" --timeout 60 --memory-size 1024 \
    --environment "Variables={ANTHROPIC_API_KEY=${ANTHROPIC},OPENAI_API_KEY=${OPENAI}}" >/dev/null
fi
aws lambda wait function-updated --function-name "${FN}" --region "${REGION}"

# --- 4. API Gateway ---------------------------------------------------------
echo ">> Ensuring API Gateway..."
API_ID=$(aws apigateway get-rest-apis --region "${REGION}" \
  --query "items[?name=='${FN}-api'].id | [0]" --output text 2>/dev/null || true)

if [[ -z "${API_ID}" || "${API_ID}" == "None" ]]; then
  API_ID=$(aws apigateway create-rest-api --name "${FN}-api" \
    --description "LangGraph RAG agent" --region "${REGION}" \
    --query 'id' --output text)

  ROOT_ID=$(aws apigateway get-resources --rest-api-id "${API_ID}" --region "${REGION}" \
    --query 'items[?path==`/`].id' --output text)
  RESOURCE_ID=$(aws apigateway create-resource --rest-api-id "${API_ID}" \
    --parent-id "${ROOT_ID}" --path-part query --region "${REGION}" \
    --query 'id' --output text)
  aws apigateway put-method --rest-api-id "${API_ID}" --resource-id "${RESOURCE_ID}" \
    --http-method POST --authorization-type NONE --api-key-required \
    --region "${REGION}" >/dev/null
  aws apigateway put-integration --rest-api-id "${API_ID}" --resource-id "${RESOURCE_ID}" \
    --http-method POST --type AWS_PROXY --integration-http-method POST \
    --uri "arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${LAMBDA_ARN}/invocations" \
    --region "${REGION}" >/dev/null
  aws lambda add-permission --function-name "${FN}" --statement-id "apigw-${API_ID}" \
    --action lambda:InvokeFunction --principal apigateway.amazonaws.com \
    --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT}:${API_ID}/*/POST/query" \
    --region "${REGION}" >/dev/null
fi

echo ">> Deploying to 'prod' stage..."
aws apigateway create-deployment --rest-api-id "${API_ID}" --stage-name prod \
  --region "${REGION}" >/dev/null

# --- 5. API key + usage plan ------------------------------------------------
echo ">> Ensuring API key + usage plan..."
KEY_ID=$(aws apigateway get-api-keys --region "${REGION}" \
  --query "items[?name=='${KEY_NAME}'].id | [0]" --output text 2>/dev/null || true)
if [[ -z "${KEY_ID}" || "${KEY_ID}" == "None" ]]; then
  KEY_ID=$(aws apigateway create-api-key --name "${KEY_NAME}" --enabled \
    --region "${REGION}" --query 'id' --output text)
fi
KEY_VALUE=$(aws apigateway get-api-key --api-key "${KEY_ID}" --include-value \
  --region "${REGION}" --query 'value' --output text)

PLAN_ID=$(aws apigateway get-usage-plans --region "${REGION}" \
  --query "items[?name=='${PLAN_NAME}'].id | [0]" --output text 2>/dev/null || true)
if [[ -z "${PLAN_ID}" || "${PLAN_ID}" == "None" ]]; then
  PLAN_ID=$(aws apigateway create-usage-plan --name "${PLAN_NAME}" \
    --throttle "burstLimit=${BURST_LIMIT},rateLimit=${RATE_LIMIT}" \
    --quota "limit=${DAILY_QUOTA},period=DAY" \
    --api-stages "apiId=${API_ID},stage=prod" \
    --region "${REGION}" --query 'id' --output text)
  aws apigateway create-usage-plan-key --usage-plan-id "${PLAN_ID}" \
    --key-id "${KEY_ID}" --key-type API_KEY --region "${REGION}" >/dev/null
fi

URL="https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod/query"

# Persist URL and KEY in .env (overwrite previous lines).
TMP=$(mktemp)
grep -v -E '^LAMBDA_API_(URL|KEY)=' .env > "${TMP}" || true
mv "${TMP}" .env
echo "LAMBDA_API_URL=${URL}" >> .env
echo "LAMBDA_API_KEY=${KEY_VALUE}" >> .env

echo ""
echo "========================================="
echo "Deploy complete."
echo "URL:    ${URL}"
echo "Key:    saved to .env (last 4: ...${KEY_VALUE: -4})"
echo "Quota:  ${DAILY_QUOTA}/day, ${RATE_LIMIT} rps (burst ${BURST_LIMIT})"
echo "========================================="
