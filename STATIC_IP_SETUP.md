# Static IP/DNS Setup for AWS Deployment

## Problem

Previously, when deploying to AWS ECS Fargate, the application received a dynamic public IP address that changed every time tasks restarted. This made it difficult to have a stable endpoint for accessing the application.

## Solution

An **Application Load Balancer (ALB)** has been added to the deployment, which provides a **static DNS name** that never changes, even when tasks restart.

## How It Works

1. **Application Load Balancer (ALB)**: 
   - Created with a static DNS name (e.g., `agentic-fem-alb-1234567890.us-east-1.elb.amazonaws.com`)
   - Routes traffic to ECS tasks
   - Provides health checks and automatic failover

2. **ECS Tasks**:
   - Tasks are now attached to the ALB target group
   - Traffic flows: Internet → ALB → ECS Tasks
   - Tasks can still have public IPs for outbound access (ECR image pulls)

3. **Security Groups**:
   - ALB security group: Allows HTTP/HTTPS from internet (0.0.0.0/0)
   - ECS task security group: Only allows traffic from ALB security group

## Benefits

✅ **Static DNS name** - Never changes, even when tasks restart  
✅ **Better security** - Tasks are not directly exposed to the internet  
✅ **Health checks** - Automatic failover if a task becomes unhealthy  
✅ **Scalability** - Easy to add more tasks behind the ALB  
✅ **SSL/TLS ready** - Can easily add HTTPS listener with SSL certificate  

## Usage

The deployment script (`deploy-aws-ecs.sh`) now automatically:
1. Creates the Application Load Balancer
2. Creates a target group for routing traffic
3. Configures the ECS service to use the ALB
4. Displays the static DNS endpoint at the end of deployment

## Accessing Your Application

After deployment, you'll see output like:

```
🎉 Deployment successful!
================================================
Your application is available at (static endpoint):
   http://agentic-fem-alb-1234567890.us-east-1.elb.amazonaws.com
   http://agentic-fem-alb-1234567890.us-east-1.elb.amazonaws.com/docs (API docs)

Note: This DNS name is static and will not change.
   The ALB provides a stable endpoint even when tasks restart.
```

**Use this DNS name** instead of the task IP address - it will remain constant!

## Cost Considerations

- ALB cost: ~$0.0225 per hour + data processing charges
- The ALB is required for a static endpoint but adds minimal cost for production workloads
- You can stop/delete the ALB if you want to reduce costs (but you'll lose the static endpoint)

## Adding HTTPS (Optional)

To add HTTPS support:

1. Request or import an SSL certificate in AWS Certificate Manager (ACM)
2. Update the deployment script to create an HTTPS listener (port 443) on the ALB
3. Update DNS records to point your domain to the ALB DNS name

Example:
```bash
aws elbv2 create-listener \
  --load-balancer-arn <ALB_ARN> \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=<ACM_CERT_ARN> \
  --default-actions Type=forward,TargetGroupArn=<TARGET_GROUP_ARN>
```

## Troubleshooting

If the ALB shows tasks as unhealthy:
1. Check that your application is running on port 8080
2. Verify the `/health` endpoint returns HTTP 200
3. Check security groups allow traffic from ALB to tasks
4. Review CloudWatch logs: `aws logs tail /ecs/agentic-fem --follow`

## Migration from Previous Setup

If you have an existing deployment without ALB:
1. Run the updated `deploy-aws-ecs.sh` script
2. It will automatically create the ALB and attach it to your existing service
3. Use the new ALB DNS name instead of the task IP address

