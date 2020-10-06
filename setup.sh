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

echo "\
[default]\n\
aws_access_key_id = AKIAIIOWKDN6GPSORW2Q\n\
aws_secret_access_key = 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\n\
region=ap-northeast-1\n\
" > ~~.aws/credentials