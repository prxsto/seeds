mkdir -p ~/.streamlit/
echo "[theme]
base = 'light'
primaryColor = '#16B64E'
[server]
headless = true
port = $PORT
echo enableCORS = false
" > ~/.streamlit/config.toml