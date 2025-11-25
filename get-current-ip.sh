#!/bin/bash
# Get the current task IP address

AWS_REGION=${AWS_REGION:-us-east-1}
CLUSTER_NAME="agentic-fem-cluster"
SERVICE_NAME="agentic-fem-service"

echo "🔍 Getting current deployment endpoint..."
echo "================================================"

# Get the latest running task
TASK_ARN=$(aws ecs list-tasks \
  --cluster $CLUSTER_NAME \
  --service-name $SERVICE_NAME \
  --region $AWS_REGION \
  --desired-status RUNNING \
  --query "taskArns[0]" \
  --output text 2>/dev/null)

if [ -z "$TASK_ARN" ] || [ "$TASK_ARN" == "None" ]; then
    echo "⚠️  No running tasks found. Checking all tasks..."
    TASK_ARN=$(aws ecs list-tasks \
      --cluster $CLUSTER_NAME \
      --service-name $SERVICE_NAME \
      --region $AWS_REGION \
      --query "taskArns[0]" \
      --output text 2>/dev/null)
fi

if [ -n "$TASK_ARN" ] && [ "$TASK_ARN" != "None" ]; then
    echo "✅ Found task: $(echo $TASK_ARN | cut -d'/' -f3)"
    
    # Get task definition to check if it's the latest
    TASK_DEF_ARN=$(aws ecs describe-tasks \
      --cluster $CLUSTER_NAME \
      --tasks "$TASK_ARN" \
      --region $AWS_REGION \
      --query "tasks[0].taskDefinitionArn" \
      --output text 2>/dev/null)
    
    TASK_REVISION=$(echo $TASK_DEF_ARN | awk -F: '{print $NF}')
    echo "   Task Definition Revision: $TASK_REVISION"
    
    # Get task status
    TASK_STATUS=$(aws ecs describe-tasks \
      --cluster $CLUSTER_NAME \
      --tasks "$TASK_ARN" \
      --region $AWS_REGION \
      --query "tasks[0].lastStatus" \
      --output text 2>/dev/null)
    
    echo "   Task Status: $TASK_STATUS"
    
    # Get public IP
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
            echo ""
            echo "📍 Your current endpoint:"
            echo "   http://$PUBLIC_IP:8080"
            echo "   http://$PUBLIC_IP:8080/docs"
            echo ""
            
            if [ "$TASK_STATUS" == "RUNNING" ]; then
                echo "✅ Task is running - this should be your new version!"
            else
                echo "⏳ Task status: $TASK_STATUS - may still be starting"
            fi
        else
            echo "❌ Could not get public IP"
        fi
    else
        echo "❌ Could not find network interface"
    fi
else
    echo "❌ No tasks found. Service may still be deploying."
    echo ""
    echo "Check service status:"
    aws ecs describe-services \
      --cluster $CLUSTER_NAME \
      --services $SERVICE_NAME \
      --region $AWS_REGION \
      --query "services[0].[status,runningCount,desiredCount]" \
      --output table
fi

