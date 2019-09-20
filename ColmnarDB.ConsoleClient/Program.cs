using System;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Timers;
using ColmnarDB.NetworkClient;

namespace ColmnarDB.ConsoleClient
{
    public class Program
    {
        public static string ipAddress = "127.0.0.1";
        public static short port = 12345;
        private static bool exit = false;
        private static ColumnarDBClient client;
        private static Mutex mutex;
        /// <summary>
        /// Reads input from console
        /// commands:
        /// use [database]
        /// u
        /// [query]
        /// help
        /// h
        /// exit
        /// quit
        /// q
        /// timing
        /// t
        /// docs
        /// man
        /// </summary>
        public static void Main(string[] args)
        {
            int timeout = 30000;
            if (args.Length >= 2)
            {
                if (args[0] == "-t")
                {
                    timeout = Convert.ToInt32(args[1]);
                    Console.WriteLine("Set timeout to: " + timeout.ToString());
                }
                if (args[0] == "-h")
                {
                    if (args[1].Contains(':'))
                    {
                        ipAddress = args[1].Split(':')[0];
                        port = Convert.ToInt16(args[1].Split(':')[1]);
                    }
                    else
                    {
                        ipAddress = args[1];
                    }
                }
            }

            client = new ColumnarDBClient("Host=" + ipAddress + ";" + "Port=" + port.ToString() + ";");
            client.Connect();
            var heartBeatTimer = new System.Timers.Timer(timeout);
            heartBeatTimer.Elapsed += HeartBeatTimer_Elapsed;
            heartBeatTimer.AutoReset = true;
            heartBeatTimer.Enabled = true;
            UseDatabase use = new UseDatabase();
            ImportCSV import = new ImportCSV(ipAddress, port);
            Query query = new Query();
            mutex = new Mutex();
            ReadLine.HistoryEnabled = true;

            while (!exit)
            {
                string wholeCommand = ReadLine.Read("> ");
                string[] splitCommand = wholeCommand.Split(" ");

                string command = splitCommand[0].ToLower();
                string parameters = "";

                if (command != "docs" && command != "man" && command != "t" && command != "timing" && command != "q" && command != "quit" && command != "exit" && command != "u" && command != "use" && command != "h" && command != "help"  && command != "import" )
                {
                    parameters = wholeCommand;
                    parameters = parameters.Trim();
                    parameters = parameters[parameters.Length - 1] != ';' ? parameters + ';' : parameters;
                    command = "query";
                }
                else if (splitCommand.Length > 1)
                {
                    parameters = wholeCommand.Substring(command.Length + 1);
                    parameters = parameters.Trim();
                }
                mutex.WaitOne();
                switch (command)
                {
                    case "exit":
                    case "quit":
                    case "q":
                        exit = true;
                        client.Close();
                        break;

                    case "u":
                    case "use":
                        if (parameters == "")
                        {
                            Console.WriteLine("Missing argument");
                            break;
                        }
                        parameters = parameters[parameters.Length - 1] == ';' ? parameters.Substring(0, parameters.Length - 1) : parameters;
                        use.Use(parameters, client);

                        break;
                    case "query":
                        if (parameters == "")
                        {
                            Console.WriteLine("Missing argument");
                            break;
                        }

                        query.RunQuery(parameters, Console.WindowWidth, client);

                        break;

                    case "t":
                    case "timing":
                        if (parameters == "")
                        {
                            Console.WriteLine("Missing argument");
                            break;
                        }

                        query.RunTestQuery(parameters, Console.WindowWidth, client);

                        break;

                     case "import":
                        
                        string[] splitParameters = Regex.Matches(parameters, @"[\""].+?[\""]|[^ ]+").Cast<Match>().Select(m => m.Value).ToArray();

                        if (parameters == "" || splitParameters.Length < 2)
                        {
                            Console.WriteLine("Missing arguments");
                            break;
                        }

                        string database = splitParameters[0];
                        string filePath = splitParameters[1];
                        if (filePath.Length > 0 && filePath.ElementAt(0) == '\"')
                        {
                            filePath = filePath.Substring(1, filePath.Length - 2);
                        }

                        if (splitParameters.Length == 2)
                        {
                            import.Import(filePath, database);
                        }
                        else if (splitParameters.Length == 3) {
                            import.Import(filePath, database, bool.Parse(splitParameters[2]));
                        }
                        else if (splitParameters.Length == 4)
                        {
                            import.Import(filePath, database, bool.Parse(splitParameters[2]), columnSeparator: splitParameters[3].ElementAt(0));
                        }
                        else if (splitParameters.Length == 5)
                        {
                            import.Import(filePath, database, bool.Parse(splitParameters[2]), columnSeparator: splitParameters[3].ElementAt(0), threadsCount: int.Parse(splitParameters[4]));
                        }
                        else if (splitParameters.Length == 6)
                        {
                            import.Import(filePath, database, bool.Parse(splitParameters[2]), columnSeparator: splitParameters[3].ElementAt(0), threadsCount: int.Parse(splitParameters[4]), batchSize: int.Parse(splitParameters[5]));
                        }

                        break; 

                    case "h":
                    case "help":
                        //formated console output
                        const string format = "{0,-30} {1,-30}";

                        Console.WriteLine();
                        Console.WriteLine(String.Format(format, "h, help", "Show information about commands"));
                        Console.WriteLine(String.Format(format, "docs, man", "Prints 'Documentation is available at https://docs.tellstory.ai/'"));
                        Console.WriteLine(String.Format(format, "u [database], use [database]", "Set current working database"));
                        Console.WriteLine(String.Format(format, "[query]", "Run given query"));
                        Console.WriteLine(String.Format(format, "t [query], timing [query]", "Run a query " + Query.numberOfQueryExec + 1 + " times and print the first and average cached execution time."));
                        Console.WriteLine(String.Format(format, "q, quit, exit", "Exit the console"));

                        Console.WriteLine();
                        break;
                    case "docs":
                    case "man":
                        Console.WriteLine("Documentation is available at https://docs.tellstory.ai/");
                        break;
                    default:
                        Console.WriteLine("Unknown command, for more information about commands type 'help'");
                        break;
                }
                mutex.ReleaseMutex();
            }
            heartBeatTimer.Stop();
            heartBeatTimer.Dispose();
        }

        private static void HeartBeatTimer_Elapsed(object sender, ElapsedEventArgs e)
        {
            try
            {
                mutex.WaitOne();
                client.Heartbeat();
            }
            finally
            {
                mutex.ReleaseMutex();
            }
        }
    }
}
