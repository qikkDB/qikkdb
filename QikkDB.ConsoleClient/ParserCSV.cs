using QikkDB.Types;
using CsvHelper;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;

namespace QikkDB.ConsoleClient
{
    public class ParserCSV
    {
        public class Configuration
        {
            public char ColumnSeparator { get; private set; }
            public Encoding Encoding { get; private set; }
            public CultureInfo CultureInfo { get; private set; }
            public int BatchSize { get; private set; }
            public bool HasHeader { get; private set; }

            public Configuration(int batchSize = 0, bool hasHeader = true, Encoding encoding = null, char columnSeparator = char.MinValue, CultureInfo cultureInfo = null)
            {
                this.HasHeader = hasHeader;

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

        public class ParserException : Exception
        {
            public ParserException(string message) :
                base(message)
            {
            }
        }

        public static char[] reservedCharacters = { '/', '\\', '?', '%', '*', ':', '|', '"', '<', '>', ' ', '@' };
        public static char[] commonSeparators = { ',', ';', '|', '\t', '\\', '/', ':' };

        private Configuration configuration;
        private string[] header;
        private Dictionary<string, Type> types;
        private StreamReader streamReader;
        private NetworkClient.ColumnarDataTable tableData;
        private IParser dataParser;        
        private bool finished = false;
        long startBytePosition;
        long endBytePosition;

        public ParserCSV(string tableName, StreamReader streamReader, Dictionary<string, Type> types = null, Configuration configuration = null, long startBytePosition = 0, long endBytePosition = 0)
        {
            if (configuration != null)
            {
                this.configuration = configuration;
            }
            else
            {
                this.configuration = new Configuration();
            }

            this.streamReader = streamReader;
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
                if (streamReader.BaseStream is FileStream)
                {
                    string file = (streamReader.BaseStream as FileStream).Name;
                    this.types = GuessTypes(file, configuration.HasHeader, configuration.ColumnSeparator, configuration.Encoding);
                }                
            }
            this.header = new string[this.types.Keys.Count];
            this.types.Keys.CopyTo(this.header, 0);

            this.tableData = new NetworkClient.ColumnarDataTable(tableName);
            foreach (var head in this.header)
            {
                tableData.AddColumn(head, this.types[head]);
            }

            dataParser = new CsvReader(this.streamReader).Parser;
            dataParser.Configuration.Delimiter = this.configuration.ColumnSeparator.ToString();
            dataParser.Configuration.BadDataFound = null;
            dataParser.Configuration.TrimOptions = CsvHelper.Configuration.TrimOptions.Trim;
            // skip header at the beginning of the file or 
            // also at the beginning of thread chunk
            if (startBytePosition == 0 && this.configuration.HasHeader)
            {
                dataParser.Read();                
            }
            else if (startBytePosition > 0)
            {
                dataParser.Read();                
            }

        }

        public static Encoding GuessEncoding(string file)
        {
            if (!File.Exists(file))
            {
                throw new ParserException("Could not find file " + file + ". Check if the path or filename is correct.");
            }

            Encoding result = Encoding.UTF8;
            try
            {
                using (var reader = new StreamReader(file, true))
                {
                    reader.Peek();
                    result = reader.CurrentEncoding;
                }
            }
            catch
            {
                throw new ParserException("Could not access file when getting encoding. Maybe the file is corrupted or opened by another program.");
            }
            return result;
        }

        public static char GuessSeparator(string file, Encoding encoding)
        {
            if (!File.Exists(file))
            {
                throw new ParserException("Could not find file " + file + ". Check if the path or filename is correct.");
            }

            char result = ',';
            string sample = "";

            try
            {
                var stream = File.Open(file, FileMode.Open, FileAccess.Read, FileShare.Read);

                using (var reader = new StreamReader(stream, encoding))
                {
                    sample = reader.ReadLine();

                    // if the first line is almost the whole file, it is wrong csv
                    if (sample.Length >= 0.95 * (reader.BaseStream.Length))
                    {
                        throw new ParserException("The file seems to have only header or does not follow CSV structure.");
                    }
                }

                var histogram = sample.GroupBy(c => c).ToDictionary(g => g.Key, g => g.Count()).OrderByDescending(x => x.Value).ToDictionary(x => x.Key, x => x.Value);

                foreach (var value in histogram.Keys)
                {
                    if (commonSeparators.Contains(value))
                    {
                        result = value;
                        break;
                    }
                }

                // if not common separator (everything is one column), then heuristics is applied
                if (sample.Split(result).Length == 1)
                {
                    foreach (var pair in histogram)
                    {
                        if (!Char.IsLetter(pair.Key) && !Char.IsDigit(pair.Key))
                        {
                            if (pair.Value > 0.2 * (sample.Length) && pair.Value < 0.5 * (sample.Length))
                            {
                                result = pair.Key;
                            }
                        }
                    }
                }

                stream.Dispose();
            }
            catch
            {
                throw new ParserException("Error occured while guessing separator. Maybe the file is corrupted or does not follow CSV structure.");
            }

            return result;
        }

        public static Dictionary<string, Type> GuessTypes(string file, bool hasHeader, char separator, Encoding encoding)
        {
            if (!File.Exists(file))
            {
                throw new ParserException("Could not find file " + file + ". Check if the path or filename is correct.");
            }

            Dictionary<string, Type> result = new Dictionary<string, Type>();
            string[] header;
            
            List<string[]> rows = new List<string[]>();
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
            var stream = File.Open(file, FileMode.Open, FileAccess.Read, FileShare.Read);
            var parser = new CsvReader(new StreamReader(stream, encoding)).Parser;
            parser.Configuration.Delimiter = separator.ToString();
            parser.Configuration.BadDataFound = null;
            parser.Configuration.TrimOptions = CsvHelper.Configuration.TrimOptions.Trim;

            // Read CSV header
            if (hasHeader)
            {
                var row = parser.Read();
                header = new string[row.Length];
                var headerSet = new HashSet<string>();
                for (int i = 0; i < row.Length; i++)
                {
                    // if contains column name, append "2", "3", ...
                    string colName = row[i];
                    int suffixId = 2;
                    while (headerSet.Contains(colName))
                    {
                        colName = row[i] + "-" + suffixId++;
                    }
                    headerSet.Add(colName);
                    header[i] = colName;                    
                }
            }
            else
            {
                // if there is no header, name of column is C0, C1...
                var row = parser.Read();
                header = new string[row.Length];
                for (int i=0; i<row.Length;i++)
                {
                    header[i] = "C" + i;
                }
                
            }

            if (header.Length == 0)
            {
                throw new ParserException("Could not extract header. Maybe separator (" + separator + ") was wrongly guessed or first line of the file is not following CSV structure.");
            }

            for (int i = 0; i < header.Length; i++)
            {
                foreach (var reservedCharacter in reservedCharacters)
                {
                    header[i] = header[i].Replace(reservedCharacter, '_');
                }
            }

            Dictionary<string, List<string>> topRows = new Dictionary<string, List<string>>();
            foreach (var head in header)
            {
                topRows.Add(head, new List<string>());
            }
            for (int i = 0; i < 10000; i++)
            {
                string[] vals = null;
                try
                {
                    vals = parser.Read();
                }
                catch
                {
                    continue;
                }
                if (vals == null)
                    break;

                if (vals.Length != header.Length)
                {
                    continue;
                }

                rows.Add(vals);
                int columnIndex = 0;
                foreach (var val in vals)
                {
                    string key = header[columnIndex];
                    topRows[key].Add(vals[columnIndex]);                    
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

        private static Type GuessTableColumnType(string columnName, List<string> topNvalues)
        {
            bool boolError = false;
            bool singleError = false;
            bool doubleError = false;
            bool longError = false;
            bool intError = false;
            bool datetimeError = false;
            bool complexPolygonError = false;
            bool pointError = false;
            foreach (var value in topNvalues)
            {
                if (value == "" || value.ToLower() == "null" || value.ToLower() == "nan")
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

                if (!bool.TryParse(value, out bool boolRes))
                {
                    boolError = true;
                }

                try
                {
                    // Short string are definitely not points
                    if (value.Length < 10 || !value.Contains("POINT"))
                    {
                        pointError = true;
                    }
                    else
                    {
                        Point point = new Point(value);
                    }                    
                }
                catch
                {
                    pointError = true;
                }

                try
                {
                    // Short string are definitely not polygons
                    if (value.Length < 12 || !value.Contains("POLYGON"))
                    {
                        complexPolygonError = true;
                    }
                    else
                    {
                        ComplexPolygon polygon = new ComplexPolygon(value);
                    }
                }
                catch
                {
                    complexPolygonError = true;
                }

                if (doubleError && longError && intError && datetimeError && singleError && boolError && complexPolygonError && pointError) break;
            }
            if (!intError) return typeof(Int32);
            if (!longError) return typeof(Int64);
            if (!singleError) return typeof(Single);
            if (!doubleError) return typeof(Double);
            if (!datetimeError) return typeof(DateTime);
            if (!boolError) return typeof(bool);
            if (!complexPolygonError) return typeof(ComplexPolygon);
            if (!pointError) return typeof(Point);
            return typeof(String);
        }

        public NetworkClient.ColumnarDataTable GetNextParsedDataBatch(out long importedLinesCount, out long errorLinesCount)
        {
            importedLinesCount = 0;
            errorLinesCount = 0;

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
            
            while (((batchSize > 0 && readLinesCount < batchSize) || batchSize == 0) && ((this.startBytePosition + dataParser.Context.CharPosition) < this.endBytePosition))
            {
                string[] row = null;
                try
                {
                    row = dataParser.Read();
                }
                catch
                {
                    errorCount++;
                    continue;
                }

                if (row == null)
                    break;

                for (int i = 0; i < row.Length; i++)
                {
                    if (row[i].Length == 0 || (row[i].Length == 4 && row[i].ToLower() == "null"))
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
                                else if (type == typeof(bool))
                                    convertedRow[i] = Convert.ToByte(bool.Parse(row[i]));
                                else if (type == typeof(Point))
                                    convertedRow[i] = new Point(row[i]);
                                else if (type == typeof(ComplexPolygon))
                                    convertedRow[i] = new ComplexPolygon(row[i]);
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
                errorLinesCount = errorCount;
            }
            if (readLinesCount == 0)
            {
                finished = true;
                dataParser.Dispose();
                return null;
            }
            importedLinesCount = tableData.GetSize();
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
