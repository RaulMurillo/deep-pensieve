import sys
import os
import time

send_email = True

# if __name__ == '__main__':
if(send_email):
    PACKAGE_PARENT = '..'
    SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
    sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

    from other.mail import send_mail

    s = int(8)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)

    subject = 'CNN Training compleated'
    body = 'The training phase with data type X on TensorFlow (X) has finished after X h, min, sec!\n\nThe Top-5 is X and training history is:\nX'

    path = os.path.abspath('../other/credentials.txt')

    send_mail(subject=subject, mail_body=body, credentials=path)
