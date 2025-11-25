# Docker Structure

## Overview

This document describes the Docker structure for the Agentic FEM application, including the Dockerfile, docker-compose configuration, and containerization strategy.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                          │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   Base Image │  │  Dependencies │  │  Application │     │
│  │  (Mambaforge)│  │  (Conda/Pip) │  │  (Python)    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│         │                  │                  │             │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Host System                                    │
│              Port 8080                                      │
│              Volume Mounts                                  │
└─────────────────────────────────────────────────────────────┘
```

## Base Image

### Mambaforge

**Image**: `condaforge/mambaforge:latest`

**Purpose**: Provides conda/mamba package manager and Python environment

**Benefits**:
- Fast package installation with mamba
- Pre-configured Python environment
- Access to conda-forge channel
- Better dependency resolution

## Dockerfile Structure

### Stage 1: System Dependencies

```dockerfile
FROM condaforge/mambaforge:latest
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libxft2 \
    libxrender1 \
    libfontconfig1 \
    libxext6 \
    libxinerama1 \
    && rm -rf /var/lib/apt/lists/*
```

**Purpose**: Install system-level dependencies required by the application

**Dependencies**:
- `build-essential`: Compilation tools
- `git`: Version control
- `curl`: HTTP client (for health checks)
- X11 libraries: Required by GMSH (even in headless mode)

### Stage 2: Conda Dependencies

```dockerfile
# Install core dependencies via mamba/conda
RUN mamba install -y -c conda-forge \
    python=3.10 \
    nodejs \
    fenics-dolfinx \
    mpich \
    petsc \
    gmsh \
    openblas \
    numpy \
    scipy \
    matplotlib \
    pandas \
    h5py \
    && mamba clean -afy
```

**Purpose**: Install scientific computing dependencies via conda

**Key Packages**:
- `python=3.10`: Python runtime
- `nodejs`: Node.js runtime (for frontend)
- `fenics-dolfinx`: FEniCS finite element library
- `mpich`: MPI implementation
- `petsc`: Linear algebra library
- `gmsh`: Mesh generation library
- `openblas`: BLAS/LAPACK implementation
- Scientific Python stack: numpy, scipy, matplotlib, pandas, h5py

### Stage 3: Python Dependencies

```dockerfile
# Copy requirements file and install remaining Python packages via pip
COPY requirements.txt /app/requirements.txt
RUN grep -vE "^(gmsh|numpy|scipy|matplotlib|pandas|h5py)" requirements.txt > /tmp/requirements_filtered.txt && \
    pip install --no-cache-dir -r /tmp/requirements_filtered.txt && \
    rm /tmp/requirements_filtered.txt
```

**Purpose**: Install Python packages not available in conda-forge

**Strategy**:
- Filter out packages already installed via conda
- Prevent binary incompatibilities between conda and pip
- Install remaining packages via pip

### Stage 4: Node.js Dependencies

```dockerfile
# Copy package files for Node.js dependencies
COPY package.json package-lock.json* /app/

# Install Node.js dependencies
RUN if [ -f package-lock.json ]; then \
        npm ci --silent; \
    else \
        npm install --silent; \
    fi
```

**Purpose**: Install Node.js dependencies (VTK.js)

**Strategy**:
- Use `npm ci` if package-lock.json exists (deterministic install)
- Use `npm install` otherwise

### Stage 5: Application Code

```dockerfile
# Copy the entire application
COPY . /app/

# Set up entrypoint script
RUN chmod +x /app/docker-entrypoint.sh && \
    head -n 1 /app/docker-entrypoint.sh | grep -q "^#!/bin/bash"

# Create necessary directories
RUN mkdir -p /app/frontend/static/css /app/frontend/static/js /app/frontend/templates

# Copy VTK.js to static directory
RUN if [ -d "/app/node_modules/vtk.js" ]; then \
        cp node_modules/vtk.js/vtk.js frontend/static/js/vtk.js && \
        cp node_modules/vtk.js/vtk.js.map frontend/static/js/vtk.js.map || true; \
    fi
```

**Purpose**: Copy application code and set up static files

**Steps**:
1. Copy application code
2. Set up entrypoint script
3. Create necessary directories
4. Copy VTK.js to static directory

### Stage 6: Environment Configuration

```dockerfile
# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV DISPLAY=
ENV GMSH_NO_DISPLAY=1

# Expose the application port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
```

**Purpose**: Configure environment and health checks

**Environment Variables**:
- `PYTHONUNBUFFERED=1`: Unbuffered Python output
- `PYTHONPATH=/app`: Python path
- `DISPLAY=`: Empty display (headless mode)
- `GMSH_NO_DISPLAY=1`: GMSH headless mode

**Health Check**:
- Interval: 30 seconds
- Timeout: 10 seconds
- Start period: 40 seconds
- Retries: 3
- Command: HTTP GET on `/health` endpoint

### Stage 7: Entrypoint

```dockerfile
# Use entrypoint script to handle .env file automatically
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Run the application
CMD ["python", "-m", "uvicorn", "apps.main_app:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Purpose**: Set up entrypoint and default command

**Entrypoint Script**: Handles .env file setup and application startup

## Docker Compose Configuration

### Service Definition

```yaml
version: '3.8'

services:
  agentic-fem:
    build:
      context: .
      dockerfile: Dockerfile
    image: agentic-fem:latest
    ports:
      - "8080:8080"
    volumes:
      - ./nlp_parser/src/.env:/app/nlp_parser/src/.env
    restart: unless-stopped
```

**Configuration**:
- **Build**: Build from Dockerfile in current directory
- **Image**: `agentic-fem:latest`
- **Ports**: Map host port 8080 to container port 8080
- **Volumes**: Mount .env file for configuration
- **Restart**: Restart unless stopped

### Volume Mounts

**Purpose**: Mount configuration files and data

**Mounts**:
- `.env` file: OpenAI API key configuration
- Optional: Data directories for persistence

## Entrypoint Script

### Purpose

The entrypoint script (`docker-entrypoint.sh`) handles:
1. Creating .env file from environment variables
2. Checking for mounted .env file
3. Running the application

### Script Structure

```bash
#!/bin/bash
# Agentic FEM Docker Entrypoint Script

# Create .env file from OPENAI_API_KEY if set
if [ -n "$OPENAI_API_KEY" ]; then
    echo "OPENAI_API_KEY=$OPENAI_API_KEY" > /app/nlp_parser/src/.env
fi

# Check for mounted .env file
if [ -f /app/nlp_parser/src/.env ]; then
    echo "Using .env file from volume mount"
else
    echo "Warning: No .env file found"
fi

# Run the application
exec "$@"
```

## Building the Image

### Local Build

```bash
docker build -t agentic-fem:latest .
```

### Platform-Specific Build

```bash
# For AWS ECS (linux/amd64)
docker build --platform linux/amd64 -t agentic-fem:latest .
```

### Build with Cache

```bash
docker build --cache-from agentic-fem:latest -t agentic-fem:latest .
```

## Running the Container

### Docker Run

```bash
docker run -d \
  --name agentic-fem \
  -p 8080:8080 \
  -v $(pwd)/nlp_parser/src/.env:/app/nlp_parser/src/.env \
  agentic-fem:latest
```

### Docker Compose

```bash
docker-compose up -d
```

### With Environment Variables

```bash
docker run -d \
  --name agentic-fem \
  -p 8080:8080 \
  -e OPENAI_API_KEY=your-api-key \
  agentic-fem:latest
```

## Health Checks

### Container Health Check

The Dockerfile includes a health check:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1
```

### Manual Health Check

```bash
curl http://localhost:8080/health
```

### Health Check Status

```bash
docker ps
# Check HEALTH STATUS column
```

## Logging

### View Logs

```bash
# Docker run
docker logs agentic-fem

# Docker compose
docker-compose logs -f agentic-fem
```

### Log Rotation

Configure log rotation in `daemon.json`:

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  }
}
```

## Troubleshooting

### Common Issues

1. **Build fails**: Check Docker version and system requirements
2. **Container fails to start**: Check logs and environment variables
3. **Health check fails**: Verify application is running on port 8080
4. **GMSH errors**: Verify X11 libraries are installed
5. **Permission errors**: Check file permissions and user/group

### Debugging

1. **Interactive shell**:
   ```bash
   docker run -it --entrypoint /bin/bash agentic-fem:latest
   ```

2. **Check environment**:
   ```bash
   docker exec agentic-fem env
   ```

3. **Check processes**:
   ```bash
   docker exec agentic-fem ps aux
   ```

4. **Check logs**:
   ```bash
   docker logs agentic-fem
   ```

## Optimization

### Image Size

- Use multi-stage builds (already implemented)
- Remove unnecessary files
- Clean package caches
- Use .dockerignore to exclude files

### Build Time

- Use build cache
- Order Dockerfile instructions by change frequency
- Use parallel builds where possible

### Runtime Performance

- Use appropriate resource limits
- Monitor resource usage
- Optimize application code
- Use health checks for automatic recovery

## Security

### Best Practices

1. **Use minimal base images**: Use specific tags, not `latest`
2. **Scan images**: Use Docker security scanning
3. **Limit capabilities**: Use `--cap-drop` and `--cap-add`
4. **Non-root user**: Run container as non-root user (if possible)
5. **Secrets management**: Use secrets management, not environment variables
6. **Network security**: Use Docker networks to isolate containers

### Image Scanning

```bash
# Use Trivy
trivy image agentic-fem:latest

# Use Docker Scout
docker scout cves agentic-fem:latest
```

## Files

- `Dockerfile`: Docker image definition
- `docker-compose.yml`: Docker Compose configuration
- `docker-entrypoint.sh`: Entrypoint script
- `.dockerignore`: Files to exclude from build context
- `requirements.txt`: Python dependencies
- `package.json`: Node.js dependencies

## Production Considerations

### Resource Limits

Set resource limits in docker-compose.yml:

```yaml
services:
  agentic-fem:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Networking

Use Docker networks for isolation:

```yaml
services:
  agentic-fem:
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

### Persistence

Use volumes for data persistence:

```yaml
services:
  agentic-fem:
    volumes:
      - app-data:/app/data

volumes:
  app-data:
```

### Monitoring

Use monitoring tools:
- Prometheus for metrics
- Grafana for visualization
- ELK stack for logging

