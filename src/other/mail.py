import smtplib
import json
from email.message import EmailMessage


def send_mail(subject, mail_body, credentials, send_to=[]):

    with open(credentials, 'r') as f:
        d = json.load(f)

    gmail_user = d['user']
    gmail_password = d['password']

    sent_from = gmail_user
    if not send_to: # Auto e-mail
        send_to = [gmail_user]
    to = send_to

    # Create the message
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['To'] = ', '.join(to)
    msg['From'] = sent_from
    msg.set_content(mail_body)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        # server.sendmail(from_addr=sent_from, to_addrs=to, msg=email_text)
        server.send_message(msg)
        server.close()

        print('Email sent!')
    except:
        print('Something went wrong...')
 