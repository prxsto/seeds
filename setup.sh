mkdir -p ~/.streamlit/
echo "[theme]" > ~/.streamlit/config.toml
echo "base = 'light'" > ~/.streamlit/config.toml
echo "primaryColor = '#16B64E'" > ~/.streamlit/config.toml
echo "[server]" > ~/.streamlit/config.toml
echo "headless = true" > ~/.streamlit/config.toml
echo "port = $PORT" > ~/.streamlit/config.toml
echo "enableCORS = false" > ~/.streamlit/config.toml