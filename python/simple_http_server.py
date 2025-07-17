import socketserver
import http.server
import mimetypes

import os
import sys
import argparse

class FileMan:
    def __init__(self, in_file):
        self.file = in_file
        self.handler = None

    def __del__(self):
        if self.handler:
            self.handler.close()

    def readbinary(self):
        self.handler = open(self.file, 'rb')
        return self.handler.read()

    def read(self):
        self.handler = open(self.file, 'r', encoding='utf8')
        return self.handler.read()

class HtmlHandler(http.server.BaseHTTPRequestHandler):
    def __init__(self, request, client_address, server, *args, **kwargs):
        if not self.html_dir:
            raise RuntimeError("you should set 'html_dir' first for HtmlHandler though 'set_cofiguration' first")
        self.server = server
        super().__init__(request, client_address, server, *args, **kwargs)

    @classmethod
    def set_cofiguration(cls, html_dir):
        cls.html_dir = html_dir


    def send_binary_file(self, path):
        candidate = os.path.abspath(os.path.join(self.html_dir, path))
        if os.path.isfile(candidate) and os.path.exists(candidate):
            self.send_response(200)
            self.send_header("Content-type", mimetypes.guess_type(candidate)[0] or "text/plain")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            self.end_headers()
            self.wfile.write(FileMan(candidate).readbinary())
        else:
            self.send_response(403)


    def do_GET(self):
        print (self.path)
        if self.path == "/exit" or self.path == "/exit/":
            self.server.shutdown()
            sys.exit()
        elif self.path == "/":
            self.send_binary_file("index.html")
        else:
            self.send_binary_file(self.path[1:])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Simple Server',
                    description = 'Simple file server on python',
                    epilog = 'made by idimus')

    parser.add_argument('-r', '--resources_dir', help='root directory for resources', default=".")
    parser.add_argument('-p', '--port', help='server serving port', default=8000, type=int)
    
    args = parser.parse_args()

    HtmlHandler.set_cofiguration (html_dir = args.resources_dir)

    with socketserver.ThreadingTCPServer(("", args.port), HtmlHandler) as httpd:
        httpd.allow_reuse_address = True
        httpd.daemon_threads = True
        print ("serving at port", args.port)

        httpd.serve_forever()
