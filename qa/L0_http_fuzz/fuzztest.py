import unittest
import sqlite3
from boofuzz import *
import glob
import os


class FuzzTest(unittest.TestCase):
    def _run_fuzz(self):
        session = Session(target=Target(
            connection=TCPSocketConnection("127.0.0.1", 8000)),
            keep_web_open=False)

        s_initialize(name="Request")
        with s_block("Request-Line"):
            s_group("Method", ["GET", "HEAD", "POST", "PUT",
                               "DELETE", "CONNECT", "OPTIONS", "TRACE"])
            s_delim(" ", name="space-1")
            s_group("Request-URI", ["/v2/models/simple", "/v2/models/simple/infer",
                                    "/v2/models/simple/config", "/v2/models/simple/stats",
                                    "/v2/health/ready", "/v2/health/live"])
            s_delim(" ", name="space-2")
            s_string("HTTP/1.1", name="HTTP-Version")
            s_static("\r\n", name="Request-Line-CRLF")
            s_string("Host:", name="Host-Line")
            s_delim(" ", name="space-3")
            s_static("\r\n", name="Host-Line-CRLF")
        s_static("\r\n", "Request-CRLF")

        session.connect(s_get("Request"))
        session.fuzz()

    def test_failures_from_db(self):
        self._run_fuzz()

        # Get latest db file
        files = glob.glob('boofuzz-results/*')
        dbfile = max(files, key=os.path.getctime)

        conn = sqlite3.connect(dbfile)
        c = conn.cursor()

        # Get number of failures, should be 0
        self.assertEqual(len([x for x in c.execute(
            "SELECT * FROM steps WHERE type=\"fail\"")]), 0)


if __name__ == "__main__":
    unittest.main()
