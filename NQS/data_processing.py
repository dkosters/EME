import os
import os.path
import pandas as pd
import numpy as np
import argparse

def main(FLAGS):
    print(FLAGS.filepath)
    if os.path.isfile(FLAGS.filepath):
        joblist = pd.read_csv(FLAGS.filepath, sep=',|;', engine="python")
    else:
        print("no job file found")
        exit()

    df_joblist = pd.DataFrame(joblist)
    df = []
    df_plus = []
    for i, id in enumerate(df_joblist["jobid"].tolist()):
        os.system("eacct -j {0} -c pre_results.csv >/dev/null".format(id))
        os.system("eacct -j {0} -l -c pre_results_plus.csv >/dev/null".format(id))
        pre_results = pd.read_csv("pre_results.csv", sep=',|;', engine="python")
        pre_results_plus = pd.read_csv("pre_results_plus.csv", sep=',|;', engine="python") 
        df.append(pd.DataFrame(pre_results).iloc[-1:])
        df_plus.append(pd.DataFrame(pre_results_plus).iloc[-1:])
        print("read %s" % (id))
        os.system("rm pre_results.csv")
        os.system("rm pre_results_plus.csv")
    
    df_results = pd.concat(df, ignore_index=True)
    df_results_plus = pd.concat(df_plus,  ignore_index=True)
    print(df_results)
    print(df_joblist)
    for name, values in df_joblist.iteritems():
        df_results[name] = values
        df_results_plus[name] = values

    outputfilepath="earoutput/"+os.path.basename(FLAGS.filepath)
    print(outputfilepath)
    df_results.to_csv(outputfilepath, index=False, header=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default="measurements.csv", help="the joblist file should contain a list of jobids other information also useful and will be added")
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
