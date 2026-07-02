# V22 Gold Scalping Bot + Web Dashboard
# Deploy on Contabo VPS via Coolify
FROM python:3.11-slim

WORKDIR /app

# Install system deps + MT5 wine layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget xvfb wine wine64 x11vnc xdotool procps \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY trading_bot/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir flask gunicorn

# Copy bot code
COPY trading_bot/ trading_bot/
COPY scripts/ scripts/
COPY logs/ logs/

# Expose web port
EXPOSE 5000

# Start both bot + web server
CMD python -m trading_bot.main_v22 & gunicorn -w 1 -b 0.0.0.0:5000 trading_bot.web_server:app