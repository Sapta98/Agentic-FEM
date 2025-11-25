#!/bin/bash
# Quick script to find your endpoint - tries multiple methods

AWS_REGION=${AWS_REGION:-us-east-1}

echo "🔍 Finding your deployment endpoint..."
echo "================================================"

# Method 1: Try to find ALB
echo -e "\n1️⃣ Checking for Application Load Balancer..."
ALB_DNS=$(aws elbv2 describe-load-balancers \
  --region $AWS_REGION \
  --query "LoadBalancers[?contains(LoadBalancerName, 'agentic') || contains(LoadBalancerName, 'fem')].DNSName" \
  --output text 2>/dev/null | head -1)

if [ -n "$ALB_DNS" ] && [ "$ALB_DNS" != "None" ]; then
    echo "✅ Found ALB!"
    echo ""
    echo "📍 Your endpoint (static DNS):"
    echo "   http://$ALB_DNS"
    echo "   http://$ALB_DNS/docs"
    exit 0
fi

# Method 2: Check all ALBs
echo "Checking all ALBs in region..."
ALL_ALBS=$(aws elbv2 describe-load-balancers \
  --region $AWS_REGION \
  --query "LoadBalancers[*].[LoadBalancerName,DNSName]" \
  --output table 2>/dev/null)

if [ -n "$ALL_ALBS" ]; then
    echo "$ALL_ALBS"
fi

# Method 3: Check ECS service for task IP
echo -e "\n2️⃣ Checking ECS tasks for direct IP..."
for CLUSTER in agentic-fem-cluster; do
    for SERVICE in agentic-fem-service; do
        TASK_ARN=$(aws ecs list-tasks \
          --cluster $CLUSTER \
          --service-name $SERVICE \
          --region $AWS_REGION \
          --query "taskArns[0]" \
          --output text 2>/dev/null || echo "")
        
        if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ]; then
            echo "Found task: $TASK_ARN"
            
            ENI_ID=$(aws ecs describe-tasks \
              --cluster $CLUSTER \
              --tasks "$TASK_ARN" \
              --region $AWS_REGION \
              --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" \
              --output text 2>/dev/null || echo "")
            
            if [ -n "$ENI_ID" ] && [ "$ENI_ID" != "None" ]; then
                PUBLIC_IP=$(aws ec2 describe-network-interfaces \
                  --network-interface-ids "$ENI_ID" \
                  --region $AWS_REGION \
                  --query "NetworkInterfaces[0].Association.PublicIp" \
                  --output text 2>/dev/null || echo "")
                
                if [ -n "$PUBLIC_IP" ] && [ "$PUBLIC_IP" != "None" ]; then
                    echo "✅ Found task IP!"
                    echo ""
                    echo "📍 Your endpoint (dynamic IP):"
                    echo "   http://$PUBLIC_IP:8080"
                    echo "   http://$PUBLIC_IP:8080/docs"
                    echo ""
                    echo "⚠️  Note: This IP will change when tasks restart"
                    exit 0
                fi
            fi
        fi
    done
done

# Method 4: List all resources
echo -e "\n3️⃣ Listing all resources..."
echo ""
echo "ECS Clusters:"
aws ecs list-clusters --region $AWS_REGION --output text 2>/dev/null || echo "  (none found)"
echo ""
echo "Load Balancers:"
aws elbv2 describe-load-balancers --region $AWS_REGION --query "LoadBalancers[*].LoadBalancerName" --output text 2>/dev/null || echo "  (none found)"

echo -e "\n💡 Tip: The endpoint should have been shown at the end of your deployment script."
echo "   If you scrolled up, check the deployment output for 'Deployment successful!'"

