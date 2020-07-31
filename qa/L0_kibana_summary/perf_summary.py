from email import encoders
from fpdf import FPDF
import summarize_kibana as sk
from datetime import date
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
import glob

pdf = FPDF()
today = date.today().strftime("%Y-%m-%d")
pdf.set_font('Arial', size=12)
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

            pdf.add_page()
            pdf.cell(200, 12, txt=title, align='C')
            pdf.image(title + ".png", x=20, y=20, w=125, h=100)
pdf.output("summary_" + today + ".pdf", "F")

FROM = "hemantj@nvidia.com"
TO = 'sw-dl-triton@exchange.nvidia.com'
SUBJECT = "Triton Performance Summary :"+ today
msg = MIMEMultipart()
msg['Subject'] = SUBJECT
msg['From'] = FROM
msg['To'] = TO

files = glob.glob("*.csv") + ["summary_" + today + ".pdf"]
for filename in files:
    attachment = open(filename, "rb")
    p = MIMEBase('application', 'octet-stream')
    p.set_payload((attachment).read())
    encoders.encode_base64(p)
    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(p)

mailServer = smtplib.SMTP("mailgw.nvidia.com")
mailServer.sendmail(FROM, TO, msg.as_string())
mailServer.quit()
