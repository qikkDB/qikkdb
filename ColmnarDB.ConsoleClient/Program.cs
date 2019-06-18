using System;
using ColmnarDB.NetworkClient;

namespace ColmnarDB.ConsoleClient
{
    public class Program
    {
        public static readonly string IpAddress = "127.0.0.1";
        public static readonly short Port = 12345;

        /// <summary>
        /// Reads input from console
        /// commands:
        /// use [database]
        /// query [query]
        /// import [file path]
        /// help
        /// exit
        /// </summary>
        public static void Main(string[] args)
        {
            ColumnarDBClient client = new ColumnarDBClient(IpAddress,Port);
            client.Connect();
            
            UseDatabase use = new UseDatabase();
            ImportCSV import = new ImportCSV();
            Query query = new Query();
            
            bool exit = false;

            while (!exit)
            {
                // wholeCommand = command + parameters
                Console.Out.Write("> ");
                string wholeCommand = Console.ReadLine();
                string[] splitCommand = wholeCommand.Split(" ");

                string command = splitCommand[0];
                string parameters = "";
                string database = "";
                string filePath = "";

                // get parameters
                if (splitCommand.Length > 1) 
                {
                    parameters = wholeCommand.Substring(command.Length + 1);
                }

                switch (command)
                {
                    case "exit":                       
                        exit = true;
                        client.Close();
                        break;
                    case "use":
                        if (parameters == "")
                        {
                            Console.WriteLine("Missing argument");
                            break;
                        }
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
                    case "import":

                        string[] splitParameters = parameters.Split(" ");

                        if (parameters == "" || splitParameters.Length < 2)
                        {
                            Console.WriteLine("Missing arguments");
                            break;
                        }

                        database = splitParameters[0];
                        filePath = parameters.Substring(database.Length + 1);

                        import.Import(filePath,database, client);
                       
                        break;
                    case "help":
                        //formated console output
                        const string format = "{0,-30} {1,-30}";

                        Console.WriteLine();
                        Console.WriteLine(String.Format(format, "use [database]", "Set current working database"));
                        Console.WriteLine(String.Format(format, "query [query]", "Run given query"));
                        Console.WriteLine(String.Format(format, "import [database] [file path]", "Import given .csv file into database"));
                        Console.WriteLine(String.Format(format, "help", "Show information about commands"));
                        Console.WriteLine(String.Format(format, "exit", "Exit the console"));
                        Console.WriteLine();
                        break;
                    default:
                        Console.WriteLine("Unknown command, for more information about commands type 'help'");
                        break;

                }
            }

        }
        
    }
}
