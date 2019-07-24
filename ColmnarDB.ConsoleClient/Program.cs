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
        /// [query]
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
                string wholeCommand = ConsoleExtension.ReadConsole(">");
                string[] splitCommand = wholeCommand.Split(" ");

                string command = splitCommand[0];
                string parameters = "";
                string database = "";
                string filePath = "";

                if (command != "exit" && command != "use" && command != "import" && command != "help")
                {
                    parameters = wholeCommand;
                    command = "query";
                } 
                else if (splitCommand.Length > 1) 
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
                        Console.WriteLine(String.Format(format, "[query]", "Run given query"));
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

    public class ConsoleExtension
    {
        private static Queue<string> InputQueue = new Queue<string>();

        public static string ReadConsole(string Prompt)
        {
            string CommandStringBuilder = "";
            int CounterOfQueue = 0;
 
            while (true)
            {
                // loop until Enter key is pressed
                ConsoleKeyInfo KeyInfoPressed = Console.ReadKey();
                switch (KeyInfoPressed.Key)
                {
                    case ConsoleKey.UpArrow:
                        // present the member (from end) in the queue not further then the queue size
                        if (CounterOfQueue == 0)
                        {
                            CounterOfQueue = InputQueue.Count;
                            CommandStringBuilder = InputQueue.ElementAt(CounterOfQueue - 1);
                            ClearConsole(Prompt, CommandStringBuilder);
                            CounterOfQueue--;
                        }
                        else
                        {
                            if (CounterOfQueue == 0)
                            {
                                CounterOfQueue = 0;
                                ClearConsole(Prompt, "");
                                CommandStringBuilder = "";
                            }
                            else
                            {
                                CommandStringBuilder = InputQueue.ElementAt(CounterOfQueue - 1);
                                ClearConsole(Prompt, CommandStringBuilder);
                                CounterOfQueue--;
                            }
                        }
                        break;
 
                    case ConsoleKey.DownArrow:
                        // present the member (from begin) in the queue not further then the queue size
                        if (InputQueue.Count > CounterOfQueue)
                        {
                            CommandStringBuilder = InputQueue.ElementAt(CounterOfQueue);
                            ClearConsole(Prompt, CommandStringBuilder);
                            CounterOfQueue++;
                        }
                        else
                        {
                            CounterOfQueue = 0;
                            ClearConsole(Prompt, "");
                            CommandStringBuilder = "";
                        }
                        break;
 
                    case ConsoleKey.LeftArrow:
                        // move the cursor to the correct position, not further then the end of the question
                        if (Console.CursorLeft > Prompt.Length)
                        {
                            Console.SetCursorPosition(Console.CursorLeft - 2, Console.CursorTop);
                        }
                        break;
 
                    case ConsoleKey.RightArrow:
                        // move the cursor to the correct position, not further then the end of the buffer
                        if (Console.CursorLeft < Console.BufferWidth)
                        {
                            Console.SetCursorPosition(Console.CursorLeft + 1, Console.CursorTop);
                        }
                        break;
 
                    case ConsoleKey.Backspace:
                        // Backspace, remove the char until we reach the question, we can't delete the question
                        if (Console.CursorLeft > Prompt.Length - 1)
                        {
                            Console.Write(new string(' ', 1));
                            Console.SetCursorPosition(Console.CursorLeft - 1, Console.CursorTop);
                            CommandStringBuilder = CommandStringBuilder.Remove(CommandStringBuilder.Length - 1, 1);
                        }
                        else
                        {
                            Console.SetCursorPosition(Prompt.Length, Console.CursorTop);
                        }
                        break;
 
                    case ConsoleKey.Delete:
                        break;
 
                    default:
                        // just add the char to the answer building string
                        CommandStringBuilder = CommandStringBuilder + KeyInfoPressed.KeyChar.ToString();
                        ClearConsole(Prompt, CommandStringBuilder);
                        break;
 
                    case ConsoleKey.Enter:
                        // exit this routine and return the Answer to process further
                        return CommandStringBuilder;
                }
            }
        }

        private static void ClearConsole(string Question, string Answer)
        {
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(new String(' ', Console.BufferWidth - Question.Length));
            Console.SetCursorPosition(0, Console.CursorTop);
            Console.Write(Question + Answer);
        }
    }
}
