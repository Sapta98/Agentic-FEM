#!/bin/bash
# Simple script to show endpoint - runs in default region

AWS_REGION=${AWS_REGION:-us-east-1}

echo "🔍 Finding your endpoint in region: $AWS_REGION"
echo "================================================"

# Method 1: ALB DNS (preferred - static)
echo -e "\n📡 Method 1: Application Load Balancer (static DNS)..."
aws elbv2 describe-load-balancers \
  --region $AWS_REGION \
  --output json 2>/dev/null | \
  jq -r '.LoadBalancers[] | select(.LoadBalancerName | contains("agentic") or contains("fem")) | "✅ ALB Found!\n   Name: \(.LoadBalancerName)\n   DNS: \(.DNSName)\n   Status: \(.State.Code)\n\n📍 Your endpoint:\n   http://\(.DNSName)\n   http://\(.DNSName)/docs"' 2>/dev/null

# Method 2: Direct task IP
echo -e "\n📡 Method 2: ECS Task Direct IP..."
CLUSTER_NAME="agentic-fem-cluster"
SERVICE_NAME="agentic-fem-service"

TASK_ARN=$(aws ecs list-tasks \
  --cluster $CLUSTER_NAME \
  --service-name $SERVICE_NAME \
  --region $AWS_REGION \
  --query "taskArns[0]" \
  --output text 2>/dev/null)

if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ]; then
    echo "Task found: $TASK_ARN"
    
    ENI_ID=$(aws ecs describe-tasks \
      --cluster $CLUSTER_NAME \
      --tasks "$TASK_ARN" \
      --region $AWS_REGION \
      --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
      --output text 2>/dev/null)
    
    if [ -n "$ENI_ID" ] && [ "$ENI_ID" != "None" ]; then
        PUBLIC_IP=$(aws ec2 describe-network-interfaces \
          --network-interface-ids "$ENI_ID" \
          --region $AWS_REGION \
          --query "NetworkInterfaces[0].Association.PublicIp" \
          --output text 2>/dev/null)
        
        if [ -n "$PUBLIC_IP" ] && [ "$PUBLIC_IP" != "None" ]; then
            echo -e "✅ Task IP Found!\n"
            echo "📍 Your endpoint (dynamic IP - may change):"
            echo "   http://$PUBLIC_IP:8080"
            echo "   http://$PUBLIC_IP:8080/docs"
        fi
    fi
else
    echo "No running tasks found"
fi

# Method 3: List all resources
echo -e "\n📋 Available resources:"
echo ""
echo "ECS Clusters:"
aws ecs list-clusters --region $AWS_REGION --output text 2>/dev/null | head -5 || echo "  (none found)"

echo ""
echo "Load Balancers:"
aws elbv2 describe-load-balancers --region $AWS_REGION --query "LoadBalancers[*].[LoadBalancerName,DNSName]" --output table 2>/dev/null | head -10 || echo "  (none found)"

echo -e "\n💡 If you just deployed, check your terminal output above"
echo "   Look for: '🎉 Deployment successful!'"

