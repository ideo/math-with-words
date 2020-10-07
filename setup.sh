mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"jgambino@ideo.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

mkdir -p ~/.aws/

echo "\
[default]\n\
aws_access_key_id = \"$AWS_KEY\"\n\
aws_secret_access_key = \"$AWS_SECRET\"\n\
region=\"$AWS_REGION\"\n\
" > ~/.aws/credentials