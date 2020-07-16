from elasticsearch import Elasticsearch, RequestsHttpConnection
import json

class QueryKibana():
    def __init__(self):
        host = "https://gpuwa.nvidia.com/elasticsearch"
        timeout = 1000
        self.index = "df-dlfw-trtis-benchmarks-*"
        self.es = Elasticsearch(timeout=timeout, hosts=host, connection_class=RequestsHttpConnection)
 
    def fetch_results(self, value_list, where_dict, limit=1):
        """Runs a query to fetch data given a where condition and the names of fields asked for.
        The values are ordered in desc order of '@timestamp'.

        Parameters
        ----------
        value_list : list
            The list of fields whose values are needed.
        where_dict : dict
            The mapping of of fields and values to match them with.
        limit : int
            The max size of the values to return. Defaults to 1.
            Use 0 for no limit.

        Returns
        -------
        results : dict
            The list of lists of the required fields in the specified order.

        """
        # Check if index exists
        if not self.es.indices.exists(index=self.index):
            print("Index " + self.index + " not exists")
            exit()

        where_str = ""
        for key in where_dict:
            if where_str != "":
                where_str += " AND "
            if key.startswith("s_"):
                where_str += key + " LIKE \'" + where_dict[key] + "\'"
            else:
                where_str += key + "=" + where_dict[key]

        query_str = "SELECT " + str(value_list).strip("[]").replace("\'", "")+" FROM \"" + self.index + \
                "\" WHERE (" + where_str + ") ORDER BY \'@timestamp\' DESC"
        if limit:
            query_str += " LIMIT " + str(limit)
        body = {"query": query_str}

        results = self.es.transport.perform_request(
            'POST', '/_xpack/sql', params={'format': 'json'}, body=body)
        return results['rows']
    
    def close(self):
        """Closes the current connection to Elasticsearch

        """
        self.es.transport.connection_pool.close()

# Sample 
# qk = QueryKibana()
# value_list = ["d_infer_per_sec", "d_latency_p95_ms", "l_instance_count", "s_shared_memory",
#               "s_protocol", "s_framework", "l_batch_size", "s_benchmark_repo_branch"]
# where_dict = {"s_shared_memory": "none", "s_benchmark_name": "nomodel", "s_benchmark_repo_branch":"r20.06"}
# print(qk.fetch_results(value_list, where_dict=where_dict))
# qk.close()
