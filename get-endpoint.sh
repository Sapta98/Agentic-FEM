#!/bin/bash
# Helper script to find your AWS deployment endpoint
# Usage: ./get-endpoint.sh [region]

set -e

AWS_REGION=${1:-${AWS_REGION:-us-east-1}}
ALB_NAME=agentic-fem-alb
CLUSTER_NAME=agentic-fem-cluster
SERVICE_NAME=agentic-fem-service

echo "🔍 Searching for deployment in region: $AWS_REGION"
echo "================================================"

# Check if ALB exists
echo -e "\n📡 Checking for Application Load Balancer..."
ALB_DNS=$(aws elbv2 describe-load-balancers \
  --names $ALB_NAME \
  --region $AWS_REGION \
  --query "LoadBalancers[0].DNSName" \
  --output text 2>/dev/null || echo "")

if [ -n "$ALB_DNS" ] && [ "$ALB_DNS" != "None" ]; then
    echo -e "✅ Found ALB!\n"
    echo -e "📍 Your application endpoint:"
    echo "   http://$ALB_DNS"
    echo "   http://$ALB_DNS/docs (API docs)"
    exit 0
fi

# If ALB not found, check if service exists and get task IP as fallback
echo -e "⚠️  ALB not found. Checking for ECS service...\n"

SERVICE_STATUS=$(aws ecs describe-services \
  --cluster $CLUSTER_NAME \
  --services $SERVICE_NAME \
  --region $AWS_REGION \
  --query "services[0].status" \
  --output text 2>/dev/null || echo "NOT_FOUND")

if [ "$SERVICE_STATUS" != "NOT_FOUND" ] && [ "$SERVICE_STATUS" != "None" ] && [ "$SERVICE_STATUS" != "" ]; then
    echo "✅ ECS Service exists: $SERVICE_STATUS"
    
    # Get task IP
    TASK_ARN=$(aws ecs list-tasks \
      --cluster $CLUSTER_NAME \
      --service-name $SERVICE_NAME \
      --region $AWS_REGION \
      --query "taskArns[0]" \
      --output text 2>/dev/null || echo "")
    
    if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ]; then
        echo "📋 Task found: $TASK_ARN"
        
        # Get ENI and public IP
        ENI_ID=$(aws ecs describe-tasks \
          --cluster $CLUSTER_NAME \
          --tasks $TASK_ARN \
          --region $AWS_REGION \
          --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
          --output text 2>/dev/null || echo "")
        
        if [ -n "$ENI_ID" ] && [ "$ENI_ID" != "None" ]; then
            PUBLIC_IP=$(aws ec2 describe-network-interfaces \
              --network-interface-ids $ENI_ID \
              --region $AWS_REGION \
              --query "NetworkInterfaces[0].Association.PublicIp" \
              --output text 2>/dev/null || echo "")
            
            if [ -n "$PUBLIC_IP" ] && [ "$PUBLIC_IP" != "None" ]; then
                echo -e "\n📍 Your application endpoint (dynamic IP - may change):"
                echo "   http://$PUBLIC_IP:8080"
                echo "   http://$PUBLIC_IP:8080/docs (API docs)"
                echo -e "\n⚠️  Note: This IP will change when tasks restart."
                echo "   Consider running ./deploy-aws-ecs.sh to set up a static ALB endpoint."
                exit 0
            fi
        fi
    fi
    
    echo -e "\n⚠️  Service exists but no running tasks found."
    echo "   Check service status: aws ecs describe-services --cluster $CLUSTER_NAME --services $SERVICE_NAME --region $AWS_REGION"
else
    echo -e "❌ No deployment found in region: $AWS_REGION\n"
    echo "📝 To deploy your application, run:"
    echo "   ./deploy-aws-ecs.sh"
    echo ""
    echo "   Or check other regions:"
    echo "   ./get-endpoint.sh us-west-2"
    echo "   ./get-endpoint.sh eu-west-1"
fi

