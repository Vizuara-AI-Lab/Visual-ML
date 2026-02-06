@echo off
REM Visual-ML Docker Deployment Script for Windows
REM Usage: deploy.bat [dev|prod]

setlocal

set ENVIRONMENT=%1
if "%ENVIRONMENT%"=="" set ENVIRONMENT=dev

echo 🚀 Deploying Visual-ML in %ENVIRONMENT% mode...

if "%ENVIRONMENT%"=="prod" (
    set COMPOSE_FILE=docker-compose.prod.yml
    
    REM Check if .env exists
    if not exist .env (
        echo ❌ Error: .env file not found!
        echo 📝 Please create .env file from .env.docker.example
        echo    copy .env.docker.example .env
        echo    notepad .env
        exit /b 1
    )
    
    echo 📦 Building production images...
    docker-compose -f %COMPOSE_FILE% build
    
    echo 🔄 Starting production services...
    docker-compose -f %COMPOSE_FILE% up -d
    
    echo ⏳ Waiting for services to be healthy...
    timeout /t 10 /nobreak >nul
    
    echo 🗄️  Running database migrations...
    docker-compose -f %COMPOSE_FILE% exec -T backend alembic upgrade head
    
) else (
    set COMPOSE_FILE=docker-compose.yml
    
    echo 🔄 Starting development services...
    docker-compose -f %COMPOSE_FILE% up -d
    
    echo ⏳ Waiting for services to be healthy...
    timeout /t 10 /nobreak >nul
)

echo.
echo ✅ Deployment complete!
echo.
echo 📊 Service Status:
docker-compose -f %COMPOSE_FILE% ps
echo.

if "%ENVIRONMENT%"=="prod" (
    echo 🌐 Access your application:
    echo    Frontend: http://localhost
    echo    Backend API: http://localhost:8000
    echo    API Docs: http://localhost:8000/docs
) else (
    echo 🌐 Access your application:
    echo    Frontend: http://localhost:5173
    echo    Backend API: http://localhost:8000
    echo    API Docs: http://localhost:8000/docs
)

echo.
echo 📝 View logs:
echo    docker-compose -f %COMPOSE_FILE% logs -f
echo.
echo 🛑 Stop services:
echo    docker-compose -f %COMPOSE_FILE% down

endlocal
