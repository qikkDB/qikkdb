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
                for (int i = 0; i < args.Length; i += 2)
                {
                    if (args[i] == "-t")
                    {
                        timeout = Convert.ToInt32(args[i + 1]);
                        Console.WriteLine("Set timeout to: " + timeout.ToString());
                    }
                    if (args[i] == "-h")
                    {
                        if (args[i + 1].Contains(':'))
                        {
                            ipAddress = args[i + 1].Split(':')[0];
                            port = Convert.ToInt16(args[i + 1].Split(':')[1]);
                        }
                        else
                        {
                            ipAddress = args[i + 1];
                        }
                    }
                }
            }

            client = new ColumnarDBClient("Host=" + ipAddress + ";" + "Port=" + port.ToString() + ";");
            try
            {
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

                    if (wholeCommand == "")
                    {
                        continue;
                    }

                    string[] splitCommand = wholeCommand.Split(" ");

                    string command = splitCommand[0].ToLower();
                    string parameters = "";

                    if (command != "s" && command != "script" && command != "docs" && command != "man" && command != "t"
                        && command != "timing" && command != "q" && command != "quit" && command != "exit" && command != "u"
                        && command != "use" && command != "h" && command != "help" && command != "i" && command != "import")
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
                                Console.WriteLine("USE: Missing argument - database name");
                                break;
                            }
                            parameters = parameters[parameters.Length - 1] == ';' ? parameters.Substring(0, parameters.Length - 1) : parameters;
                            use.Use(parameters, client);

                            break;
                        case "query":
                            if (parameters == "")
                            {
                                Console.WriteLine("QUERY: Missing argument - query string");
                                break;
                            }

                            query.RunQuery(parameters, Console.WindowWidth, client);

                            break;

                        case "s":
                        case "script":
                            if (parameters == "")
                            {
                                Console.WriteLine("SCRIPT: Missing argument - file path");
                                break;
                            }

                            try
                            {
                                var queryFile = new System.IO.StreamReader(parameters);
                                Console.WriteLine($"Script from file path '{parameters}' has been loaded.");
                                string queryString;
                                int scriptFileLine = 0;

                                while ((queryString = queryFile.ReadLine()) != null)
                                {
                                    scriptFileLine++;

                                    // skip single line sql comments
                                    if (queryString.Trim().StartsWith("--"))
                                    {
                                        continue;
                                    }

                                    Console.WriteLine("SCRIPT: Executing query/command on line " + scriptFileLine + " from a script file: " + parameters);

                                    try
                                    {
                                        if (queryString.ToLower().StartsWith("u ") || queryString.ToLower().StartsWith("use "))
                                        {
                                            string[] splitCommand2 = queryString.Split(" ");

                                            splitCommand2[1] = splitCommand2[1][splitCommand2[1].Length - 1] == ';' ? splitCommand2[1].Substring(0, splitCommand2[1].Length - 1) : splitCommand2[1];
                                            use.Use(splitCommand2[1], client);
                                        }
                                        else
                                        {
                                            if (queryString.ToLower().StartsWith("i ") || queryString.ToLower().StartsWith("import "))
                                            {
                                                string[] splitCommand2 = queryString.Split(" ");
                                                string command2 = splitCommand2[0];
                                                string parameters2 = queryString.Substring(command2.Length + 1);
                                                string[] splitParameters2 = Regex.Matches(parameters2, @"[\""].+?[\""]|[^ ]+").Cast<Match>().Select(m => m.Value).ToArray();

                                                if (splitCommand2[1] == "" || splitParameters2.Length < 2)
                                                {
                                                    Console.WriteLine("IMPORT: Missing arguments - database name or file path");
                                                    break;
                                                }

                                                string database2 = splitParameters2[0];
                                                splitParameters2[1] = splitParameters2[1][splitParameters2[1].Length - 1] == ';' ? splitParameters2[1].Substring(0, splitParameters2[1].Length - 1) : splitParameters2[1];
                                                string filePath2 = splitParameters2[1];
                                                if (filePath2.Length > 0 && filePath2.ElementAt(0) == '\"')
                                                {
                                                    filePath2 = filePath2.Substring(1, filePath2.Length - 2);
                                                }

                                                if (splitParameters2.Length == 2)
                                                {
                                                    import.Import(filePath2, database2);
                                                }
                                                else if (splitParameters2.Length == 3)
                                                {
                                                    import.Import(filePath2, database2, int.Parse(splitParameters2[2]));
                                                }
                                                else if (splitParameters2.Length == 4)
                                                {
                                                    import.Import(filePath2, database2, int.Parse(splitParameters2[2]), bool.Parse(splitParameters2[3]));
                                                }
                                                else if (splitParameters2.Length == 5)
                                                {
                                                    import.Import(filePath2, database2, int.Parse(splitParameters2[2]), bool.Parse(splitParameters2[3]), columnSeparator: splitParameters2[4].ElementAt(0));
                                                }
                                                else if (splitParameters2.Length == 6)
                                                {
                                                    import.Import(filePath2, database2, int.Parse(splitParameters2[2]), bool.Parse(splitParameters2[3]), columnSeparator: splitParameters2[4].ElementAt(0), threadsCount: int.Parse(splitParameters2[5]));
                                                }
                                                else if (splitParameters2.Length == 7)
                                                {
                                                    import.Import(filePath2, database2, int.Parse(splitParameters2[2]), bool.Parse(splitParameters2[3]), columnSeparator: splitParameters2[4].ElementAt(0), threadsCount: int.Parse(splitParameters2[5]), batchSize: int.Parse(splitParameters2[6]));
                                                }
                                            }
                                            else
                                            {
                                                queryString = queryString.Last() != ';' ? queryString + ";" : queryString;
                                                query.RunQuery(queryString, Console.WindowWidth, client);
                                            }
                                        }  
                                    }
                                    catch (Exception e)
                                    {
                                        Console.WriteLine(e);
                                        break;
                                    } 
                                }
                            }
                            catch (System.IO.FileNotFoundException e)
                            {
                                Console.WriteLine($"File not found. File path: '{parameters}'.");
                            }

                            break;

                        case "t":
                        case "timing":
                            if (parameters == "")
                            {
                                Console.WriteLine("TIMING: Missing argument - query");
                                break;
                            }

                            query.RunTestQuery(parameters, Console.WindowWidth, client);

                            break;

                        case "i":
                        case "import":
                            string[] splitParameters = Regex.Matches(parameters, @"[\""].+?[\""]|[^ ]+").Cast<Match>().Select(m => m.Value).ToArray();

                            if (parameters == "" || splitParameters.Length < 2)
                            {
                                Console.WriteLine("IMPORT: Missing arguments - database name or file path");
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
                            else if (splitParameters.Length == 3)
                            {
                                import.Import(filePath, database, int.Parse(splitParameters[2]));
                            }
                            else if (splitParameters.Length == 4)
                            {
                                import.Import(filePath, database, int.Parse(splitParameters[2]), bool.Parse(splitParameters[3]));
                            }
                            else if (splitParameters.Length == 5)
                            {
                                import.Import(filePath, database, int.Parse(splitParameters[2]), bool.Parse(splitParameters[3]), columnSeparator: splitParameters[4].ElementAt(0));
                            }
                            else if (splitParameters.Length == 6)
                            {
                                import.Import(filePath, database, int.Parse(splitParameters[2]), bool.Parse(splitParameters[3]), columnSeparator: splitParameters[4].ElementAt(0), threadsCount: int.Parse(splitParameters[5]));
                            }
                            else if (splitParameters.Length == 7)
                            {
                                import.Import(filePath, database, int.Parse(splitParameters[2]), bool.Parse(splitParameters[3]), columnSeparator: splitParameters[4].ElementAt(0), threadsCount: int.Parse(splitParameters[5]), batchSize: int.Parse(splitParameters[6]));
                            }

                            break;

                        case "h":
                        case "help":
                            //formated console output
                            const string format = "{0,-30} {1,-30}";

                            Console.WriteLine();
                            Console.WriteLine(String.Format(format, "h, help", "Show information about commands"));
                            Console.WriteLine(String.Format(format, "docs, man", "Prints 'Documentation is available at https://docs.qikk.ly/'"));
                            Console.WriteLine(String.Format(format, "u [database], use [database]", "Set current working database"));
                            Console.WriteLine(String.Format(format, "i [path], import [path]", "Import CSV file with comma separated columns."));
                            Console.WriteLine(String.Format(format, "[query]", "Run given query"));
                            Console.WriteLine(String.Format(format, "s [file], script [file]", "Run SQL queries from a file path (also supports console client command USE)."));
                            Console.WriteLine(String.Format(format, "t [query], timing [query]", "Run a query " + Query.numberOfQueryExec + 1 + " times and print the first and average cached execution time."));
                            Console.WriteLine(String.Format(format, "q, quit, exit", "Exit the console"));

                            Console.WriteLine();
                            break;
                        case "docs":
                        case "man":
                            Console.WriteLine("Documentation is available at https://docs.qikk.ly/");
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
            catch (System.Net.Sockets.SocketException ex)
            {
                Console.WriteLine(ex.Message);
            }
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