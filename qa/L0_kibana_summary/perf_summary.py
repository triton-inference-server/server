from fpdf import FPDF
import summarize_kibana as sk
from datetime import date

pdf = FPDF()
pdf.set_font('Arial', size=12)
for protocol in ["http", "grpc"]:
    for metric in ["throughput", "latency"]:
        for payload_size in ["1", "4194304"]:
            if metric == "throughput":
                instances = "2"
            else:
                instances = "1"

            value_list = ["d_infer_per_sec", "s_framework", "\'@timestamp\'"]
            where_dict = {
                "s_shared_memory": "none",
                "s_benchmark_name": "nomodel",
                "l_size": payload_size,
                "s_protocol": protocol,
                "l_instance_count": instances
            }

            last_date = date.today().strftime("%Y-%m-%d")
            filename = metric + "_p" + payload_size + "_" + protocol + "_ma_" + last_date
            ma_df = sk.current_moving_average_dataframe(metric,
                                                        value_list,
                                                        where_dict,
                                                        last_date,
                                                        plot=True,
                                                        plot_file=filename +
                                                        ".png")
            # TODO Write to pdf instead of csv
            # ma_df.to_csv(filename + ".csv", index=False)

            pdf.add_page()
            pdf.cell(200, 12, txt=filename)
            pdf.image(filename + ".png", x=20, y=20, w=125, h=100)
pdf.output("summary_" + last_date + ".pdf", "F")
