using System;
using System.Collections.Generic;
using ColmnarDB.NetworkClient;
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

        /// <summary>
        /// Load benchmark queries from a file, execute them one by one and save results.
        /// </summary>
        public static void Main(string[] args)
        {
            ColumnarDBClient client = new ColumnarDBClient(IpAddress, Port);
            client.Connect();
            Console.Out.WriteLine("Client has successfully connected to server.");

            UseDatabase use = new UseDatabase();

            use.Use(taxiDbName, client);
            
            var file = new System.IO.StreamReader("./tempFile.txt");
            Console.Out.WriteLine("Benchmark queries from file 'tempFile.txt' were loaded.");

            string csvFileName;

            while ((csvFileName = file.ReadLine()) != null)
            {
                client.ImportCSV(taxiDbName, "testing-data/taxi-rides/" + csvFileName);
            }

            client.UseDatabase(taxiDbName);

            //test query on db if it successfully imported
            client.Query("SELECT cab_type, count(*) FROM trips GROUP BY cab_type;");

            //then press ctrl+c to kill server and so to save database
        }    
    }
}
