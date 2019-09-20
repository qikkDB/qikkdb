using CsvHelper;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

namespace TellStory.Data.Parser
{
    public class ParserCSV
    {
        public class Configuration
        {
            public char ColumnSeparator { get; private set; }
            public Encoding Encoding { get; private set; }
            public CultureInfo CultureInfo { get; private set; }
            public int BatchSize { get; private set; }

            public Configuration(int batchSize = 0, Encoding encoding = null, char columnSeparator = char.MinValue, CultureInfo cultureInfo = null)
            {
                if (encoding != null)
                {
                    this.Encoding = encoding;
                }
                else
                {
                    this.Encoding = Encoding.UTF8;
                }

                if (columnSeparator != char.MinValue)
                {
                    this.ColumnSeparator = columnSeparator;
                }
                else
                {
                    this.ColumnSeparator = ',';
                }

                if (cultureInfo != null)
                {
                    this.CultureInfo = cultureInfo;
                }
                else
                {
                    this.CultureInfo = CultureInfo.InvariantCulture;
                }

                if (batchSize != 0)
                {
                    this.BatchSize = batchSize;
                }
                else
                {
                    this.BatchSize = Int32.MaxValue;
                }
            }
        }

        private Configuration configuration;
        private string[] header;
        private Dictionary<string, Type> types;
        private string file;
        private FileStream stream;
        private StreamReader streamReader;
        private ColumnarTableData tableData;
        private IParser dataParser;        
        private bool finished = false;
        long startBytePosition;
        long endBytePosition;

        public ParserCSV(string file, Dictionary<string, Type> types = null, Configuration configuration = null, long startBytePosition = 0, long endBytePosition = 0)
        {
            if (configuration != null)
            {
                this.configuration = configuration;
            }
            else
            {
                this.configuration = new Configuration();
            }

            this.file = file;
            this.stream = File.Open(file, FileMode.Open, FileAccess.Read, FileShare.Read);
            this.streamReader = new StreamReader(this.stream, configuration.Encoding);
            this.startBytePosition = startBytePosition;
            if (endBytePosition != 0)
            {
                this.endBytePosition = endBytePosition;
            }
            else
            {
                this.endBytePosition = this.streamReader.BaseStream.Length;
            }

            this.streamReader.BaseStream.Seek(startBytePosition, SeekOrigin.Begin);            

            if (types != null)
            {
                this.types = types;
            }
            else
            {
                this.types = GuessTypes(file, configuration.ColumnSeparator, configuration.Encoding);
            }
            this.header = new string[this.types.Keys.Count];
            this.types.Keys.CopyTo(this.header, 0);

            this.tableData = new ColumnarTableData(Path.GetFileNameWithoutExtension(this.file));
            foreach (var head in this.header)
            {
                tableData.AddColumn(head, this.types[head]);
            }

            dataParser = new CsvReader(this.streamReader).Parser;
            dataParser.Configuration.Delimiter = this.configuration.ColumnSeparator.ToString();
            // skip header at the beginning of the file or at the beginning of thread chunk (this line is usually corrupted)
            dataParser.Read();
        }

        public static Encoding GuessEncoding(string file)
        {
            Encoding result = Encoding.UTF8;
            using (var reader = new StreamReader(file, true))
            {
                reader.Peek();
                result = reader.CurrentEncoding;
            }
            return result;
        }

        public static char GuessSeparator(string file, Encoding encoding)
        {
            char result = ',';
            
            var stream = File.Open(file, FileMode.Open, FileAccess.Read, FileShare.Read);
            string sample = "";
            using (var reader = new StreamReader(stream, encoding))
            {
                sample = reader.ReadLine();
            }            

            var histogram = sample.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count()).OrderBy(x => x.Value).ToDictionary(x => x.Key, x => x.Value);

            char[] commonSeparators = { ',', ';', '|', '\t', '\\', '/' };

            foreach (var value in histogram.Keys)
            {
                if (commonSeparators.Contains(value))
                {
                    result = value;
                    break;
                }
            }

            stream.Dispose();
            return result;
        }

        public static Dictionary<string, Type> GuessTypes(string file, char separator, Encoding encoding)
        {
            Dictionary<string, Type> result = new Dictionary<string, Type>();
            string[] header;
            
            List<string[]> rows = new List<string[]>();
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            var stream = File.Open(file, FileMode.Open, FileAccess.Read, FileShare.Read);
            var parser = new CsvReader(new StreamReader(stream, encoding)).Parser;
            parser.Configuration.Delimiter = separator.ToString();
            // Read CSV header
            header = parser.Read();
            Dictionary<string, string[]> topRows = new Dictionary<string, string[]>();
            foreach (var head in header)
            {
                string[] arr = new string[100];
                topRows.Add(head, arr);
            }
            for (int i = 0; i < 100; i++)
            {
                var vals = parser.Read();
                if (vals == null)
                    break;
                rows.Add(vals);
                int columnIndex = 0;
                foreach (var val in vals)
                {
                    string[] values;
                    string key = header[columnIndex];
                    topRows.TryGetValue(key, out values);
                    values[i] = vals[columnIndex];
                    topRows[key] = values;
                    columnIndex++;
                }
            }
            foreach (var head in header)
            {
                result.Add(head, GuessTableColumnType(head, topRows[head]));
            }

            parser.Dispose();
            stream.Dispose();
            return result;

        }

        private static Type GuessTableColumnType(string columnName, string[] topNvalues)
        {
            bool singleError = false;
            bool doubleError = false;
            bool longError = false;
            bool intError = false;
            bool datetimeError = false;
            foreach (var value in topNvalues)
            {
                if (value == "" || value.ToLower() == "null")
                {
                    continue;
                }

                if (!Int32.TryParse(value, out int intRes))
                {
                    intError = true;
                }
                if (!Int64.TryParse(value, out long lngRes))
                {
                    longError = true;
                }
                if (!DateTime.TryParse(value, out DateTime datetimeRes))
                {
                    datetimeError = true;
                }
                if (!Single.TryParse(value, out float sngRes))
                {
                    singleError = true;
                }
                if (!Double.TryParse(value, out double dblRes))
                {
                    doubleError = true;
                }
                if (doubleError && longError && intError && datetimeError) break;
            }
            if (!intError) return typeof(Int32);
            if (!longError) return typeof(Int64);
            if (!singleError) return typeof(Single);
            if (!doubleError) return typeof(Double);
            if (!datetimeError) return typeof(DateTime);
            return typeof(String);
        }

        public ColumnarTableData GetNextParsedDataBatch()
        {
            if (finished)
            {
                return null;
            }

            int readLinesCount = 0;
            Thread.CurrentThread.CurrentCulture = this.configuration.CultureInfo;
            int batchSize = this.configuration.BatchSize;
            
            tableData.Clear();

            int errorCount = 0;
            string errorMsg = null;
            //using (var parser = new CsvParser.CsvReader(rawData.Stream, encoding,
            // Include quotes (if exists) in result
            // default WithQuotes = false, i.e. column value "'test'" translated to 'test'
            //new CsvParser.CsvReader.Config() { WithQuotes = false, ColumnSeparator = columnSeparator }))

            while (((batchSize > 0 && readLinesCount < batchSize) || batchSize == 0) && ((this.startBytePosition + dataParser.Context.CharPosition) < this.endBytePosition))
            {
                var row = dataParser.Read();
                
                if (row == null)
                    break;

                for (int i = 0; i < row.Length; i++)
                {
                    if (row[i] == "" || row[i].ToLower() == "null")
                    {
                        row[i] = null;
                    }
                }

                try
                {
                    if (row.Length != header.Length)
                    {
                        if (readLinesCount > 0)
                        {
                            throw new Exception();
                        }
                    }
                    else
                    {
                        //tableData.AddRow(row);
                        object[] convertedRow = new object[row.Length];
                        for (int i = 0; i < row.Length; i++)
                        {
                            Type type = types[header[i]];

                            if (row[i] == null)
                            {
                                convertedRow[i] = null;
                            }
                            else
                            {
                                if (type == typeof(Int32))
                                    convertedRow[i] = Int32.Parse(row[i]);
                                else if (type == typeof(Int64))
                                    convertedRow[i] = Int64.Parse(row[i]);
                                else if (type == typeof(float))
                                    convertedRow[i] = float.Parse(row[i]);
                                else if (type == typeof(double))
                                    convertedRow[i] = double.Parse(row[i]);
                                else if (type == typeof(DateTime))
                                    convertedRow[i] = (long) (DateTime.Parse(row[i]).Subtract(new DateTime(1970, 1, 1)).TotalSeconds);
                                    //convertedRow[i] = DateTime.Parse(row[i]);
                                else if (type == typeof(string))
                                    convertedRow[i] = row[i];
                            }
                        }
                        tableData.AddRow(convertedRow);
                        readLinesCount++;
                    }                    
                }
                catch (Exception ex)
                {
                    errorCount++;
                    if (errorMsg == null) errorMsg = ex.Message;
                }
            }

            if (errorCount > 0)
            {
                //Model.Services.EventLogService.SaveLog(Model.Enums.EEventLogType.error.ToString(), new Model.Objects.Dataset(), "Error in parsing csv file. [Message: " + errorMsg + ", Failed rows: " + errorCount.ToString() + "]");
            }
            if (readLinesCount == 0)
            {
                finished = true;
                dataParser.Dispose();
                this.stream.Close();
                this.stream.Dispose();
                return null;
            }
            return tableData;
        }

        public static long GetStreamLength(string file)
        {
            long result = 0;
            using (var reader = new StreamReader(file, true))
            {
                reader.Peek();
                result = reader.BaseStream.Length;
            }
            return result;
        }

        public long GetStreamPosition()
        {
            return dataParser.Context.CharPosition;
        }
    }
}
