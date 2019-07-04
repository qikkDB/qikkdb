using System;
using System.Collections.Generic;
using ColmnarDB.NetworkClient;
using ColmnarDB.ConsoleClient;
using System.Linq;
using System.IO;

//THIS FILE IS FOR ONE TIME USAGE - TO CREATE DB FROM CSV FILES
namespace ColmnarDB.BenchmarkUtility
{
    public class TempProgram
    {
        public static readonly string IpAddress = "127.0.0.1";
        public static readonly short Port = 12345;

        public static readonly string taxiDbName = "TaxiRidesDb";
		public static readonly string csvFileName = "trips.csv";

        /// <summary>
        /// Load benchmark queries from a file, execute them one by one and save results.
        /// </summary>
        public static void Main(string[] args)
        {
            ColumnarDBClient client = new ColumnarDBClient(IpAddress, Port);
            client.Connect();

            UseDatabase use = new UseDatabase();
            ImportCSV import = new ImportCSV();
            Query query = new Query();

            import.Import("data/" + csvFileName, taxiDbName, client);

            use.Use(taxiDbName, client);

            //test query on db if it successfully imported
            query.RunQuery("SELECT cab_type, count(*) FROM trips GROUP BY cab_type;", Console.WindowWidth, client);

            //then press ctrl+c to kill server and so to save database
        }    
    }
}
