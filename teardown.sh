#!/usr/bin/env bash
# Tear down everything deploy.sh created.
# Idempotent: safe to re-run.

set -uo pipefail

REGION="${AWS_REGION:-us-east-1}"
FN="langgraph-rag-agent"
ROLE_NAME="${FN}-lambda-role"
PLAN_NAME="${FN}-plan"
KEY_NAME="${FN}-key"

ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

echo ">> Deleting usage plan + key..."
PLAN_ID=$(aws apigateway get-usage-plans --region "${REGION}" \
  --query "items[?name=='${PLAN_NAME}'].id | [0]" --output text 2>/dev/null || echo "")
KEY_ID=$(aws apigateway get-api-keys --region "${REGION}" \
  --query "items[?name=='${KEY_NAME}'].id | [0]" --output text 2>/dev/null || echo "")
if [[ -n "${PLAN_ID}" && "${PLAN_ID}" != "None" ]]; then
  if [[ -n "${KEY_ID}" && "${KEY_ID}" != "None" ]]; then
    aws apigateway delete-usage-plan-key --usage-plan-id "${PLAN_ID}" --key-id "${KEY_ID}" --region "${REGION}" || true
  fi
  aws apigateway delete-usage-plan --usage-plan-id "${PLAN_ID}" --region "${REGION}" || true
fi
if [[ -n "${KEY_ID}" && "${KEY_ID}" != "None" ]]; then
  aws apigateway delete-api-key --api-key "${KEY_ID}" --region "${REGION}" || true
fi

echo ">> Deleting REST API..."
API_ID=$(aws apigateway get-rest-apis --region "${REGION}" \
  --query "items[?name=='${FN}-api'].id | [0]" --output text 2>/dev/null || echo "")
if [[ -n "${API_ID}" && "${API_ID}" != "None" ]]; then
  aws apigateway delete-rest-api --rest-api-id "${API_ID}" --region "${REGION}" || true
fi

echo ">> Deleting Lambda function..."
aws lambda delete-function --function-name "${FN}" --region "${REGION}" 2>/dev/null || true

echo ">> Detaching + deleting role..."
aws iam detach-role-policy --role-name "${ROLE_NAME}" \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
aws iam delete-role --role-name "${ROLE_NAME}" 2>/dev/null || true

echo ">> Deleting ECR repo (force, removes images)..."
aws ecr delete-repository --repository-name "${FN}" --region "${REGION}" --force 2>/dev/null || true

echo ">> Deleting CI IAM user + policy (used by GitHub Actions)..."
CI_USER="${FN}-ci"
CI_POLICY="${FN}-ci-policy"
CI_POLICY_ARN="arn:aws:iam::${ACCOUNT}:policy/${CI_POLICY}"
# Delete access keys first
for k in $(aws iam list-access-keys --user-name "${CI_USER}" \
            --query 'AccessKeyMetadata[].AccessKeyId' --output text 2>/dev/null); do
  aws iam delete-access-key --user-name "${CI_USER}" --access-key-id "$k" 2>/dev/null || true
done
aws iam detach-user-policy --user-name "${CI_USER}" --policy-arn "${CI_POLICY_ARN}" 2>/dev/null || true
aws iam delete-policy --policy-arn "${CI_POLICY_ARN}" 2>/dev/null || true
aws iam delete-user --user-name "${CI_USER}" 2>/dev/null || true

echo ""
echo ">> Removing LAMBDA_* lines from .env..."
if [[ -f .env ]]; then
  TMP=$(mktemp)
  grep -v -E '^LAMBDA_API_(URL|KEY)=' .env > "${TMP}" || true
  mv "${TMP}" .env
fi

echo ""
echo "Teardown complete."
