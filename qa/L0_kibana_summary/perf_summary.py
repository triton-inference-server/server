import summarize_kibana as sk
from email import encoders
from datetime import date
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
import base64
import glob

html = '<html><body><a href=\"https://gpuwa.nvidia.com/kibana/app/kibana#/dashboard/ff9a1030-9a1c-11ea-8edb-c5a5e5f9de0d\">Nomodel Kibana Dashboard</a><br><center>'
today = date.today().strftime("%Y-%m-%d")
# pdf.set_font('Arial', size=12)
for metric in ["latency", "throughput"]:
    for payload_size in ["1", "4194304"]:
        for protocol in ["grpc", "http"]:
            instances = "2" if metric == "throughput" else "1"
            payload_label = "4B" if payload_size == "1" else "16MB"

            value_list = ["d_infer_per_sec", "s_framework", "\'@timestamp\'"]
            where_dict = {
                "s_shared_memory": "none",
                "s_benchmark_name": "nomodel",
                "l_size": payload_size,
                "s_protocol": protocol,
                "l_instance_count": instances
            }

            title = "Nomodel " + protocol.upper() + " " + metric + " with " + \
                payload_label + " payload"
            ma_df = sk.current_moving_average_dataframe(metric,
                                                        value_list,
                                                        where_dict,
                                                        today,
                                                        plot=True,
                                                        plot_file=title +
                                                        ".png")
            ma_df.to_csv(title + ".csv", index=False)
            img = open(title + ".png", "rb")
            data_uri = base64.b64encode(img.read()).decode('ascii')
            html += "<img src=\"data:image/png;base64,{0}\">".format(data_uri)
            html += "<div style=\"font-weight:bold\">{0}</div>".format(title)
            html += "<br><br>"

FROM = "hemantj@nvidia.com"
TO = 'sw-dl-triton@exchange.nvidia.com'
SUBJECT = "Triton Nomodel Performance Summary:" + today
msg = MIMEMultipart('alternative')
msg['Subject'] = SUBJECT
msg['From'] = FROM
msg['To'] = TO

html += '</center></body></html>'
msg.attach(MIMEText(html, "html"))

for filename in glob.glob("*.csv"):
    attachment = open(filename, "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(p)

mailServer = smtplib.SMTP("mailgw.nvidia.com")
mailServer.sendmail(FROM, TO, msg.as_string())
mailServer.quit()
