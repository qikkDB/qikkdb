using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Data.Common;
using System.Runtime.CompilerServices;
using System.Text;
using Google.Protobuf;
using Google.Protobuf.Collections;
using Google.Protobuf.WellKnownTypes;
using System.Collections;
using System.Data;
using ColmnarDB.NetworkClient.Message;
using ColmnarDB.Types;

namespace ColmnarDB.NetworkClient
{
    public class ColumnarDataTable
    {
        private List<string> columnNames;
        private Dictionary<string, IList> columnData;
        private Dictionary<string, System.Type> columnTypes;
        private int size;
        private string tableName;
        private List<string> orderedColumnNames;    //order of columns defined by SELECT

        public ColumnarDataTable(string name)
        {
            columnNames = new List<string>();
            columnData = new Dictionary<string, IList>();
            columnTypes = new Dictionary<string, System.Type>();
            orderedColumnNames = new List<string>();
            size = 0;
            this.tableName = name;
        }

        public ColumnarDataTable()
        {
            columnNames = new List<string>();
            columnData = new Dictionary<string, IList>();
            columnTypes = new Dictionary<string, System.Type>();
            orderedColumnNames = new List<string>();
            size = 0;
        }
        public ColumnarDataTable(List<string> columnNames, Dictionary<string, IList> columnData, Dictionary<string, System.Type> columnTypes, List<string> orderedColumnNames)
        {
            this.columnNames = columnNames;
            this.columnData = columnData;
            this.columnTypes = columnTypes;
            this.orderedColumnNames = orderedColumnNames;
            size = columnData.Count() > 0 ? columnData[columnNames[0]].Count : 0;
        }

        public void AddColumn(string columnName, System.Type type)
        {
            columnNames.Add(columnName);

            if (type == typeof(Int32))
                columnData.Add(columnName, new List<int?>());
            else if (type == typeof(Int64))
                columnData.Add(columnName, new List<long?>());
            else if (type == typeof(float))
                columnData.Add(columnName, new List<float?>());
            else if (type == typeof(double))
                columnData.Add(columnName, new List<double?>());
            else if (type == typeof(DateTime))
                columnData.Add(columnName, new List<long?>());
            else if (type == typeof(bool))
                columnData.Add(columnName, new List<byte?>());
            else if (type == typeof(string))
                columnData.Add(columnName, new List<String>());
            else if (type == typeof(Point))
                columnData.Add(columnName, new List<Point>());
            else if (type == typeof(ComplexPolygon))
                columnData.Add(columnName, new List<ComplexPolygon>());



            if (type == typeof(Int32))
                columnTypes.Add(columnName, typeof(Int32));
            else if (type == typeof(Int64))
                columnTypes.Add(columnName, typeof(Int64));
            else if (type == typeof(float))
                columnTypes.Add(columnName, typeof(float));
            else if (type == typeof(double))
                columnTypes.Add(columnName, typeof(double));
            else if (type == typeof(DateTime))
                columnTypes.Add(columnName, typeof(Int64));
            else if (type == typeof(bool))
                columnTypes.Add(columnName, typeof(byte));
            else if (type == typeof(string))
                columnTypes.Add(columnName, typeof(string));
            else if (type == typeof(Point))
                columnTypes.Add(columnName, typeof(Point));
            else if (type == typeof(ComplexPolygon))
                columnTypes.Add(columnName, typeof(ComplexPolygon));
        }

        public void AddRow(object[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                columnData[columnNames[i]].Add(values[i]);
            }
            size++;
        }

        public void AddRow(string[] values)
        {
            for (int i = 0; i < values.Length; i++)
            {
                System.Type type = columnTypes[columnNames[i]];
                object convertedValue = null;

                if (type == typeof(Int32))
                    convertedValue = Int32.Parse(values[i]);
                else if (type == typeof(Int64))
                    convertedValue = Int64.Parse(values[i]);
                else if (type == typeof(float))
                    convertedValue = float.Parse(values[i]);
                else if (type == typeof(double))
                    convertedValue = double.Parse(values[i]);
                else if (type == typeof(DateTime))
                    //convertedValue = (long) (DateTime.Parse(values[i]).Subtract(new DateTime(1970, 1, 1)).TotalSeconds);
                    convertedValue = DateTime.Parse(values[i]);
                else if (type == typeof(string))
                    convertedValue = values[i];
                else if (type == typeof(Point))
                    convertedValue = new Point(values[i]);
                else if (type == typeof(ComplexPolygon))
                    convertedValue = new ComplexPolygon(values[i]);

                columnData[columnNames[i]].Add(convertedValue);
            }
            size++;
        }

        public List<string> GetColumnNames()
        {
            return columnNames;
        }

        public Dictionary<string, System.Type> GetColumnTypes()
        {
            return columnTypes;
        }

        public Dictionary<string, IList> GetColumnData()
        {
            return columnData;
        }

        public List<string> GetOrderedColumnNames()
        {
            return orderedColumnNames;
        }

        public int GetSize()
        {
            return size;
        }

        public string TableName
        {
            get
            {
                return tableName;
            }
            set
            {
                tableName = value;
            }
        }

        public void Clear()
        {
            foreach (var columnName in columnNames)
            {
                columnData[columnName].Clear();
                size = 0;
            }
        }
    }
    /// <summary>
    /// Client for communication with ColumnarDB
    /// </summary>
    public class ColumnarDBClient : IDisposable
    {
        /// <summary>
        /// IP of the database server
        /// </summary>
        private readonly IPAddress _serverIP;

        /// <summary>
        /// Port of the database server
        /// </summary>
        private readonly short _serverPort;

        /// <summary>
        /// TCPClient representing connection to the database server
        /// </summary>
        private TcpClient _client;

        private const int BULK_IMPORT_FRAGMENT_SIZE = 8192 * 1024;

        /// <summary>
        /// Creates new instance of the ColumnarDBClient object
        /// </summary>
        /// <param name="connectionString">Connection string in format: Host=ipAddress;Port=port;</param>
        public ColumnarDBClient(string connectionString)
        {
            var connectionBuilder = new System.Data.Common.DbConnectionStringBuilder();
            connectionBuilder.ConnectionString = connectionString;
            string serverHostnameOrIP = (string)connectionBuilder["Host"];
            short serverPort = short.Parse((string)connectionBuilder["Port"]);

            var hostIPs = Dns.GetHostAddresses(serverHostnameOrIP);
            if (hostIPs.Length < 1)
            {
                throw new ArgumentException("Hostname given does not exist and is not an IP address");
            }
            _serverIP = hostIPs[0];
            _serverPort = serverPort;
        }


        /// <summary>
        /// Connects to the database server
        /// </summary>
        public void Connect()
        {
            _client = new TcpClient(_serverIP.AddressFamily)
            {
                ReceiveTimeout = 0,
                SendTimeout = 0
            };
            // Timeout after one minute
            _client.Connect(_serverIP, _serverPort);
            InfoMessage message = new InfoMessage { Code = InfoMessage.Types.StatusCode.ConnEstablish, Message = "" };
            try
            {
                NetworkMessage.WriteToNetwork(message, _client.GetStream());
                var responseMessage = NetworkMessage.ReadFromNetwork(_client.GetStream());
                if (!responseMessage.TryUnpack(out InfoMessage response) || response.Code != InfoMessage.Types.StatusCode.Ok)
                {
                    throw new IOException("Invalid response received from server");
                }
            }
            catch (IOException)
            {
                CloseWithoutNotify();
                throw;
            }

        }

        private T ValIfNotNulled<T>(T val, int idx, ByteString nullMask)
        {
            int shiftOffset = idx % 8;
            int byteOffset = idx / 8;
            if ((nullMask[byteOffset] >> shiftOffset & 1) == 0)
            {
                return val;
            }
            else
            {
                return default(T);
            }
        }

        private ColumnarDataTable ConvertToDictionaries(QueryResponseMessage response)
        {
            List<string> columnNames = new List<string>();
            Dictionary<string, IList> columnDatas = new Dictionary<string, IList>();
            Dictionary<string, System.Type> columnTypes = new Dictionary<string, System.Type>();
            List<string> orderedColumnNames = new List<string>();

            foreach (var column in response.ColumnOrder)
            {
                orderedColumnNames.Add(column);
            }

            foreach (var columnData in response.Payloads)
            {
                columnNames.Add(columnData.Key);
                switch (columnData.Value.PayloadCase)
                {
                    case QueryResponsePayload.PayloadOneofCase.IntPayload:
                        if (response.NullBitMasks.ContainsKey(columnData.Key))
                        {
                            columnDatas.Add(columnData.Key, new List<int?>(columnData.Value.IntPayload.IntData.ToArray().Select((val, idx) => ValIfNotNulled<int?>(val, idx, response.NullBitMasks[columnData.Key]))));
                        }
                        else
                        {
                            columnDatas.Add(columnData.Key, new List<int?>(columnData.Value.IntPayload.IntData.ToArray().Select((arg) => (int?)arg)));
                        }
                        columnTypes.Add(columnData.Key, typeof(int?));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.FloatPayload:
                        if (response.NullBitMasks.ContainsKey(columnData.Key))
                        {
                            columnDatas.Add(columnData.Key, new List<float?>(columnData.Value.FloatPayload.FloatData.ToArray().Select((val, idx) => ValIfNotNulled<float?>(val, idx, response.NullBitMasks[columnData.Key]))));
                        }
                        else
                        {
                            columnDatas.Add(columnData.Key, new List<float?>(columnData.Value.FloatPayload.FloatData.ToArray().Select((arg) => (float?)arg)));
                        }
                        columnTypes.Add(columnData.Key, typeof(float?));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.Int64Payload:
                        if (response.NullBitMasks.ContainsKey(columnData.Key))
                        {
                            columnDatas.Add(columnData.Key, new List<long?>(columnData.Value.Int64Payload.Int64Data.ToArray().Select((val, idx) => ValIfNotNulled<long?>(val, idx, response.NullBitMasks[columnData.Key]))));
                        }
                        else
                        {
                            columnDatas.Add(columnData.Key, new List<long?>(columnData.Value.Int64Payload.Int64Data.ToArray().Select((arg) => (long?)arg)));
                        }
                        columnTypes.Add(columnData.Key, typeof(long?));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.DoublePayload:
                        if (response.NullBitMasks.ContainsKey(columnData.Key))
                        {
                            columnDatas.Add(columnData.Key, new List<double?>(columnData.Value.DoublePayload.DoubleData.ToArray().Select((val, idx) => ValIfNotNulled<double?>(val, idx, response.NullBitMasks[columnData.Key]))));
                        }
                        else
                        {
                            columnDatas.Add(columnData.Key, new List<double?>(columnData.Value.DoublePayload.DoubleData.ToArray().Select((arg) => (double?)arg)));
                        }
                        columnTypes.Add(columnData.Key, typeof(double?));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.PointPayload:
                        if (response.NullBitMasks.ContainsKey(columnData.Key))
                        {
                            columnDatas.Add(columnData.Key, new List<Point>(columnData.Value.PointPayload.PointData.ToArray().Select((val, idx) => ValIfNotNulled<Point>(val, idx, response.NullBitMasks[columnData.Key]))));
                        }
                        else
                        {
                            columnDatas.Add(columnData.Key, new List<Point>(columnData.Value.PointPayload.PointData.ToArray()));
                        }
                        columnTypes.Add(columnData.Key, typeof(Point));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.PolygonPayload:
                        if (response.NullBitMasks.ContainsKey(columnData.Key))
                        {
                            columnDatas.Add(columnData.Key, new List<ComplexPolygon>(columnData.Value.PolygonPayload.PolygonData.ToArray().Select((val, idx) => ValIfNotNulled<ComplexPolygon>(val, idx, response.NullBitMasks[columnData.Key]))));
                        }
                        else
                        {
                            columnDatas.Add(columnData.Key, new List<ComplexPolygon>(columnData.Value.PolygonPayload.PolygonData.ToArray()));
                        }
                        columnTypes.Add(columnData.Key, typeof(ComplexPolygon));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.StringPayload:
                        if (response.NullBitMasks.ContainsKey(columnData.Key))
                        {
                            columnDatas.Add(columnData.Key, new List<string>(columnData.Value.StringPayload.StringData.ToArray().Select((val, idx) => ValIfNotNulled<string>(val, idx, response.NullBitMasks[columnData.Key]))));
                        }
                        else
                        {
                            columnDatas.Add(columnData.Key, new List<string>(columnData.Value.StringPayload.StringData.ToArray()));
                        }
                        columnTypes.Add(columnData.Key, typeof(string));
                        break;
                }
            }
            ColumnarDataTable ret = new ColumnarDataTable(columnNames, columnDatas, columnTypes, orderedColumnNames);
            return ret;
        }

        /// <summary>
        /// Executes given SQL query on the server
        /// </summary>
        /// <param name="query">SQL query to execute</param>
        public void Query(string query)
        {
            var message = new QueryMessage { Query = query };
            try
            {
                NetworkMessage.WriteToNetwork(message, _client.GetStream());
                var serverMessage = NetworkMessage.ReadFromNetwork(_client.GetStream());

                if (!serverMessage.TryUnpack(out InfoMessage response) || response.Code != InfoMessage.Types.StatusCode.Wait)
                {
                    throw new IOException("Invalid response received from server");
                }
                var inputMessage = NetworkMessage.ReadFromNetwork(_client.GetStream());
                if (inputMessage.TryUnpack(out InfoMessage inResponse))
                {
                    if (inResponse.Code != InfoMessage.Types.StatusCode.GetNextResult)
                    {
                        throw new QueryException(inResponse.Message);
                    }
                }
                else
                {
                    throw new IOException("Invalid response received from server");
                }

            }
            catch (IOException)
            {
                CloseWithoutNotify();
                throw;
            }

        }

        /// <summary>
        /// Get Next query result set from server
        /// </summary>
        /// <returns>Results divided by column</returns>
        public (ColumnarDataTable, Dictionary<string, float>) GetNextQueryResult()
        {
            var message = new InfoMessage { Code = InfoMessage.Types.StatusCode.GetNextResult, Message = "" };
            try
            {
                NetworkMessage.WriteToNetwork(message, _client.GetStream());
                var resultMessage = NetworkMessage.ReadFromNetwork(_client.GetStream());

                if (!resultMessage.TryUnpack(out QueryResponseMessage queryResult))
                {
                    if (resultMessage.TryUnpack(out InfoMessage errorResponse))
                    {
                        if (errorResponse.Code != InfoMessage.Types.StatusCode.Ok)
                        {
                            throw new QueryException(errorResponse.Message);
                        }
                        else
                        {
                            return (null, null);
                        }
                    }
                    throw new IOException("Invalid response received from server");
                }
                ColumnarDataTable resultSet = ConvertToDictionaries(queryResult);
                Dictionary<string, float> timingResult = new Dictionary<string, float>();
                foreach (var timing in queryResult.Timing)
                {
                    timingResult.Add(timing.Key, timing.Value);
                }
                return (resultSet, timingResult);
            }
            catch (IOException)
            {
                CloseWithoutNotify();
                throw;
            }
        }

        /// <summary>
        /// Loads a CSV file and sends it to the server for import
        /// </summary>
        /// <param name="filePath">Path to the CSV file</param>
        private void ImportCSV(string database, string filePath)
        {

            using (var fileReader = new StreamReader(filePath))
            {

                string columnNameLine = fileReader.ReadLine();
                if (fileReader.EndOfStream)
                {
                    throw new InvalidDataException("CSV file only has header");
                }

                while (!fileReader.EndOfStream)
                {
                    string payload = columnNameLine + "\n";
                    for (int i = 0; i < 1000 && !fileReader.EndOfStream; i++)
                    {
                        payload += fileReader.ReadLine() + "\n";
                    }
                    var message = new CSVImportMessage
                    { DatabaseName = database, CSVName = Path.GetFileNameWithoutExtension(filePath), Payload = payload };
                    try
                    {
                        NetworkMessage.WriteToNetwork(message, _client.GetStream());
                        var serverMessage = NetworkMessage.ReadFromNetwork(_client.GetStream());
                        if (!serverMessage.TryUnpack(out InfoMessage response))
                        {
                            throw new IOException("Invalid response received from server");
                        }
                        if (response.Code != InfoMessage.Types.StatusCode.Ok)
                        {
                            throw new QueryException(response.Message);
                        }
                    }
                    catch (IOException)
                    {
                        CloseWithoutNotify();
                        throw;
                    }
                }
            }
        }

        public void ImportCSV(string database, string tableName, string payload, List<DataType> dataTypes)
        {

            var message = new CSVImportMessage
            { DatabaseName = database, CSVName = tableName, Payload = payload };
            message.ColumnTypes.AddRange(dataTypes);
            try
            {
                NetworkMessage.WriteToNetwork(message, _client.GetStream());
                var serverMessage = NetworkMessage.ReadFromNetwork(_client.GetStream());
                if (!serverMessage.TryUnpack(out InfoMessage response))
                {
                    throw new IOException("Invalid response received from server");
                }
                if (response.Code != InfoMessage.Types.StatusCode.Ok)
                {
                    throw new QueryException(response.Message);
                }
            }
            catch (IOException)
            {
                CloseWithoutNotify();
                throw;
            }
        }

        //public void BulkImport(string tableName, Dictionary<string, string> columnTypes, Dictionary<string, IList> data)
        public void BulkImport(ColumnarDataTable dataTable)
        {
            string[] columnNames = dataTable.GetColumnNames().ToArray();
            var tableName = dataTable.TableName;
            var types = dataTable.GetColumnTypes();
            try
            {
                foreach (var column in columnNames)
                {
                    var type = types[column];
                    var columnType = DataType.ColumnInt;
                    var size = dataTable.GetSize();
                    int elementCount = size;
                    int typeSize = 1;
                    byte[] dataBuffer = null;
                    byte[] nullMask = null;
                    int i = 0;
                    switch (type.Name)
                    {
                        case nameof(Boolean):
                            {
                                columnType = DataType.ColumnInt8T;
                                dataBuffer = new byte[size];
                                var list = ((List<Boolean?>)(dataTable.GetColumnData()[column]));
                                for (i = 0; i < size; i++)
                                {
                                    if (list[i] == null)
                                    {
                                        if (nullMask == null)
                                        {
                                            int nullMaskSize = (size + sizeof(byte) * 8 - 1) / (sizeof(byte) * 8);
                                            nullMask = new byte[nullMaskSize];
                                            for (int j = 0; j < nullMaskSize; j++)
                                            {
                                                nullMask[j] = 0;
                                            }
                                        }
                                        int byteIdx = i / (sizeof(byte) * 8);
                                        int shiftIdx = i % (sizeof(byte) * 8);
                                        nullMask[byteIdx] |= (byte)(1 << shiftIdx);
                                        dataBuffer[i] = 0;
                                    }
                                    else
                                    {
                                        dataBuffer[i] = (bool)list[i] ? (byte)1 : (byte)0;
                                    }
                                }
                            }
                            break;
                        case nameof(Byte):
                            {
                                columnType = DataType.ColumnInt8T;
                                dataBuffer = new byte[size];
                                var list = ((List<byte?>)(dataTable.GetColumnData()[column]));
                                for (i = 0; i < size; i++)
                                {
                                    if (list[i] == null)
                                    {
                                        if (nullMask == null)
                                        {
                                            int nullMaskSize = (size + sizeof(byte) * 8 - 1) / (sizeof(byte) * 8);
                                            nullMask = new byte[nullMaskSize];
                                            for (int j = 0; j < nullMaskSize; j++)
                                            {
                                                nullMask[j] = 0;
                                            }
                                        }
                                        int byteIdx = i / (sizeof(byte) * 8);
                                        int shiftIdx = i % (sizeof(byte) * 8);
                                        nullMask[byteIdx] |= (byte)(1 << shiftIdx);
                                        dataBuffer[i] = 0;
                                    }
                                    else
                                    {
                                        dataBuffer[i] = (byte)list[i];
                                    }
                                }
                            }
                            break;
                        case nameof(Int32):
                            {
                                columnType = DataType.ColumnInt;
                                size *= sizeof(int);
                                typeSize = sizeof(int);
                                dataBuffer = new byte[size];
                                var list = ((List<int?>)(dataTable.GetColumnData()[column]));
                                for (i = 0; i < elementCount; i++)
                                {
                                    if (list[i] == null)
                                    {
                                        if (nullMask == null)
                                        {
                                            int nullMaskSize = (elementCount + sizeof(byte) * 8 - 1) / (sizeof(byte) * 8);
                                            nullMask = new byte[nullMaskSize];
                                            for (int j = 0; j < nullMaskSize; j++)
                                            {
                                                nullMask[j] = 0;
                                            }
                                        }
                                        int byteIdx = i / (sizeof(byte) * 8);
                                        int shiftIdx = i % (sizeof(byte) * 8);
                                        nullMask[byteIdx] |= (byte)(1 << shiftIdx);
                                        for (int j = 0; j < sizeof(int); j++)
                                        {
                                            dataBuffer[sizeof(int) * i + j] = 0;
                                        }

                                    }
                                    else
                                    {
                                        unsafe
                                        {
                                            int elem = (int)list[i];
                                            byte* elemBytes = (byte*)&elem;
                                            for (int j = 0; j < sizeof(int); j++)
                                            {
                                                dataBuffer[sizeof(int) * i + j] = elemBytes[j];
                                            }
                                        }

                                    }
                                }
                            }
                            break;
                        case nameof(Int64):
                            {
                                columnType = DataType.ColumnLong;
                                size *= sizeof(long);
                                typeSize = sizeof(long);
                                dataBuffer = new byte[size];
                                var list = ((List<long?>)(dataTable.GetColumnData()[column]));
                                for (i = 0; i < elementCount; i++)
                                {
                                    if (list[i] == null)
                                    {
                                        if (nullMask == null)
                                        {
                                            int nullMaskSize = (elementCount + sizeof(byte) * 8 - 1) / (sizeof(byte) * 8);
                                            nullMask = new byte[nullMaskSize];
                                            for (int j = 0; j < nullMaskSize; j++)
                                            {
                                                nullMask[j] = 0;
                                            }
                                        }
                                        int byteIdx = i / (sizeof(byte) * 8);
                                        int shiftIdx = i % (sizeof(byte) * 8);
                                        nullMask[byteIdx] |= (byte)(1 << shiftIdx);
                                        for (int j = 0; j < sizeof(long); j++)
                                        {
                                            dataBuffer[sizeof(long) * i + j] = 0;
                                        }

                                    }
                                    else
                                    {
                                        unsafe
                                        {
                                            long elem = (long)list[i];
                                            byte* elemBytes = (byte*)&elem;
                                            for (int j = 0; j < sizeof(long); j++)
                                            {
                                                dataBuffer[sizeof(long) * i + j] = elemBytes[j];
                                            }
                                        }

                                    }
                                }
                            }
                            break;
                        case nameof(Single):
                            {
                                columnType = DataType.ColumnFloat;
                                size *= sizeof(float);
                                typeSize = sizeof(float);
                                dataBuffer = new byte[size];
                                var list = ((List<float?>)(dataTable.GetColumnData()[column]));
                                for (i = 0; i < elementCount; i++)
                                {
                                    if (list[i] == null)
                                    {
                                        if (nullMask == null)
                                        {
                                            int nullMaskSize = (elementCount + sizeof(byte) * 8 - 1) / (sizeof(byte) * 8);
                                            nullMask = new byte[nullMaskSize];
                                            for (int j = 0; j < nullMaskSize; j++)
                                            {
                                                nullMask[j] = 0;
                                            }
                                        }
                                        int byteIdx = i / (sizeof(byte) * 8);
                                        int shiftIdx = i % (sizeof(byte) * 8);
                                        nullMask[byteIdx] |= (byte)(1 << shiftIdx);
                                        for (int j = 0; j < sizeof(float); j++)
                                        {
                                            dataBuffer[sizeof(float) * i + j] = 0;
                                        }

                                    }
                                    else
                                    {
                                        unsafe
                                        {
                                            float elem = (float)list[i];
                                            byte* elemBytes = (byte*)&elem;
                                            for (int j = 0; j < sizeof(float); j++)
                                            {
                                                dataBuffer[sizeof(float) * i + j] = elemBytes[j];
                                            }
                                        }

                                    }
                                }
                            }
                            break;
                        case nameof(Double):
                            {
                                columnType = DataType.ColumnDouble;
                                size *= sizeof(double);
                                typeSize = sizeof(double);
                                dataBuffer = new byte[size];
                                var list = ((List<double?>)(dataTable.GetColumnData()[column]));
                                for (i = 0; i < elementCount; i++)
                                {
                                    if (list[i] == null)
                                    {
                                        if (nullMask == null)
                                        {
                                            int nullMaskSize = (elementCount + sizeof(byte) * 8 - 1) / (sizeof(byte) * 8);
                                            nullMask = new byte[nullMaskSize];
                                            for (int j = 0; j < nullMaskSize; j++)
                                            {
                                                nullMask[j] = 0;
                                            }
                                        }
                                        int byteIdx = i / (sizeof(byte) * 8);
                                        int shiftIdx = i % (sizeof(byte) * 8);
                                        nullMask[byteIdx] |= (byte)(1 << shiftIdx);
                                        for (int j = 0; j < sizeof(double); j++)
                                        {
                                            dataBuffer[sizeof(double) * i + j] = 0;
                                        }

                                    }
                                    else
                                    {
                                        unsafe
                                        {
                                            float elem = (float)list[i];
                                            byte* elemBytes = (byte*)&elem;
                                            for (int j = 0; j < sizeof(double); j++)
                                            {
                                                dataBuffer[sizeof(double
                                                    ) * i + j] = elemBytes[j];
                                            }
                                        }

                                    }
                                }
                            }
                            break;
                        case nameof(Point):
                            {
                                columnType = DataType.ColumnPoint;
                                size = 0;
                                var pointList = (List<Point>)(dataTable.GetColumnData()[column]);
                                var defaultElement = new Point();
                                foreach (var elem in pointList)
                                {
                                    if (elem != null)
                                    {
                                        size += sizeof(int) + elem.CalculateSize();
                                    }
                                    else
                                    {
                                        size += sizeof(int) + defaultElement.CalculateSize();
                                    }
                                }
                                dataBuffer = new byte[size];
                                i = 0;
                                for (int j = 0; j < pointList.Count; j++)
                                {
                                    var elem = pointList[i];
                                    if (elem == null)
                                    {
                                        elem = defaultElement;
                                        if (nullMask == null)
                                        {
                                            int nullMaskSize = (elementCount + sizeof(byte) * 8 - 1) / (sizeof(byte) * 8);
                                            nullMask = new byte[nullMaskSize];
                                            for (int k = 0; k < nullMaskSize; k++)
                                            {
                                                nullMask[k] = 0;
                                            }
                                        }
                                        int byteIdx = j / (sizeof(byte) * 8);
                                        int shiftIdx = j % (sizeof(byte) * 8);
                                        nullMask[byteIdx] |= (byte)(1 << shiftIdx);
                                    }
                                    int len = elem.CalculateSize();
                                    unsafe
                                    {
                                        byte* lenBytes = (byte*)&len;
                                        for (int k = 0; k < sizeof(int); k++)
                                        {
                                            dataBuffer[i + k] = *lenBytes;
                                            lenBytes++;
                                        }
                                    }
                                    i += 4;
                                    Buffer.BlockCopy((elem).ToByteArray(), 0, dataBuffer, i, len);
                                    i += len;
                                }
                            }
                            break;
                        case nameof(ComplexPolygon):
                            {
                                columnType = DataType.ColumnPolygon;
                                size = 0;
                                var polygonList = (List<ComplexPolygon>)(dataTable.GetColumnData()[column]);
                                var defaultElement = new ComplexPolygon();
                                foreach (var elem in polygonList)
                                {
                                    if (elem != null)
                                    {
                                        size += sizeof(int) + elem.CalculateSize();
                                    }
                                    else
                                    {
                                        size += sizeof(int) + defaultElement.CalculateSize();
                                    }
                                }
                                dataBuffer = new byte[size];
                                i = 0;
                                for (int j = 0; j < polygonList.Count; j++)
                                {
                                    var elem = polygonList[j];
                                    if (elem == null)
                                    {
                                        elem = defaultElement;
                                        if (nullMask == null)
                                        {
                                            int nullMaskSize = (elementCount + sizeof(byte) * 8 - 1) / (sizeof(byte) * 8);
                                            nullMask = new byte[nullMaskSize];
                                            for (int k = 0; k < nullMaskSize; k++)
                                            {
                                                nullMask[k] = 0;
                                            }
                                        }
                                        int byteIdx = j / (sizeof(byte) * 8);
                                        int shiftIdx = j % (sizeof(byte) * 8);
                                        nullMask[byteIdx] |= (byte)(1 << shiftIdx);
                                    }
                                    int len = elem.CalculateSize();
                                    unsafe
                                    {
                                        byte* lenBytes = (byte*)&len;
                                        for (int k = 0; k < sizeof(int); k++)
                                        {
                                            dataBuffer[i + k] = *lenBytes;
                                            lenBytes++;
                                        }
                                    }
                                    i += 4;
                                    Buffer.BlockCopy((elem).ToByteArray(), 0, dataBuffer, i, len);
                                    i += len;
                                }
                            }
                            break;
                        case nameof(String):
                            {
                                columnType = DataType.ColumnString;
                                size = 0;
                                var stringList = (List<String>)(dataTable.GetColumnData()[column]);
                                var defaultElement = "";
                                foreach (var elem in stringList)
                                {
                                    if (elem != null)
                                    {
                                        size += sizeof(int) + elem.Length;
                                    }
                                    else
                                    {
                                        size += sizeof(int) + defaultElement.Length;
                                    }
                                }
                                dataBuffer = new byte[size];
                                i = 0;
                                for (int j = 0; j < stringList.Count; j++)
                                {
                                    var elem = stringList[j];
                                    if (elem == null)
                                    {
                                        elem = defaultElement;
                                        if (nullMask == null)
                                        {
                                            int nullMaskSize = (elementCount + sizeof(byte) * 8 - 1) / (sizeof(byte) * 8);
                                            nullMask = new byte[nullMaskSize];
                                            for (int k = 0; k < nullMaskSize; k++)
                                            {
                                                nullMask[k] = 0;
                                            }
                                        }
                                        int byteIdx = j / (sizeof(byte) * 8);
                                        int shiftIdx = j % (sizeof(byte) * 8);
                                        nullMask[byteIdx] |= (byte)(1 << shiftIdx);
                                    }
                                    int len = elem.Length;
                                    unsafe
                                    {
                                        byte* lenBytes = (byte*)&len;
                                        for (int k = 0; k < sizeof(int); k++)
                                        {
                                            dataBuffer[i + k] = *lenBytes;
                                            lenBytes++;
                                        }
                                    }
                                    i += 4;
                                    Buffer.BlockCopy(Encoding.UTF8.GetBytes(elem), 0, dataBuffer, i, len);
                                    i += len;
                                }
                            }
                            break;
                    }
                    int fragmentSize = 0;
                    int lastNullBuffOffset = -1;
                    for (i = 0; i < size; i += fragmentSize)
                    {
                        int elemCount = 0;
                        if (columnType == DataType.ColumnString || columnType == DataType.ColumnPolygon)
                        {
                            fragmentSize = 0;
                            while (fragmentSize < BULK_IMPORT_FRAGMENT_SIZE && i + fragmentSize < size)
                            {
                                fragmentSize += 4;
                                int strSize = 0;
                                unsafe
                                {
                                    byte* lenBytes = (byte*)&strSize;
                                    for (int k = 0; k < sizeof(int); k++)
                                    {
                                        *lenBytes = dataBuffer[i + fragmentSize - 4 + k];
                                        lenBytes++;
                                    }
                                    if (fragmentSize + strSize > BULK_IMPORT_FRAGMENT_SIZE)
                                    {
                                        fragmentSize -= 4;
                                        break;
                                    }

                                }
                                elemCount++;
                                fragmentSize += strSize;
                            }
                        }
                        else
                        {
                            fragmentSize = size - i < BULK_IMPORT_FRAGMENT_SIZE ? size - i : BULK_IMPORT_FRAGMENT_SIZE;
                            fragmentSize = (fragmentSize / typeSize) * typeSize;
                            elemCount = fragmentSize / typeSize;
                        }
                        byte[] smallBuffer = new byte[fragmentSize];
                        Buffer.BlockCopy(dataBuffer, i, smallBuffer, 0, fragmentSize);
                        int nullBuffSize = ((elemCount) + sizeof(byte) * 8 - 1) / (sizeof(byte) * 8);
                        BulkImportMessage bulkImportMessage = new BulkImportMessage()
                        { TableName = tableName, ElemCount = elemCount, ColumnName = column, ColumnType = columnType, NullMaskLen = nullMask != null ? nullBuffSize : 0 };
                        NetworkMessage.WriteToNetwork(bulkImportMessage, _client.GetStream());
                        NetworkMessage.WriteRaw(_client.GetStream(), smallBuffer, fragmentSize);
                        if (bulkImportMessage.NullMaskLen != 0)
                        {

                            int startOffset = i / (sizeof(byte) * 8);
                            if (startOffset == lastNullBuffOffset)
                            {
                                startOffset++;
                                nullBuffSize--;
                            }
                            byte[] smallNullBuffer = new byte[nullBuffSize];
                            Buffer.BlockCopy(nullMask, startOffset, smallNullBuffer, 0, nullBuffSize);
                            NetworkMessage.WriteRaw(_client.GetStream(), smallNullBuffer, nullBuffSize);
                        }
                        var serverMessage = NetworkMessage.ReadFromNetwork(_client.GetStream());
                        if (!serverMessage.TryUnpack(out InfoMessage response))
                        {
                            throw new IOException("Invalid response received from server");
                        }
                        if (response.Code != InfoMessage.Types.StatusCode.Ok)
                        {
                            throw new QueryException(response.Message);
                        }

                    }
                }
            }
            catch (IOException)
            {
                CloseWithoutNotify();
                throw;
            }
            catch (Exception ex)
            {
                CloseWithoutNotify();
                throw;
            }
        }

        /// <summary>
        /// Set currently selected database.
        /// </summary>
        /// <param name="databaseName">Name of the database to select</param>
        /// <exception cref="IOException"></exception>
        /// <exception cref="QueryException"></exception>
        public void UseDatabase(string databaseName)
        {
            var message = new SetDatabaseMessage { DatabaseName = databaseName };
            try
            {
                NetworkMessage.WriteToNetwork(message, _client.GetStream());
                var responseMessage = NetworkMessage.ReadFromNetwork(_client.GetStream());
                if (!responseMessage.TryUnpack(out InfoMessage response))
                {
                    throw new IOException("Invalid response received from server");
                }
                if (response.Code != InfoMessage.Types.StatusCode.Ok)
                {
                    throw new QueryException(response.Message);
                }
            }
            catch (IOException)
            {
                CloseWithoutNotify();
                throw;
            }

        }

        public void Heartbeat()
        {
            InfoMessage message = new InfoMessage { Code = InfoMessage.Types.StatusCode.Heartbeat, Message = "" };
            try
            {
                NetworkMessage.WriteToNetwork(message, _client.GetStream());
                var responseMessage = NetworkMessage.ReadFromNetwork(_client.GetStream());
                if (!responseMessage.TryUnpack(out InfoMessage response) || response.Code != InfoMessage.Types.StatusCode.Ok)
                {
                    throw new IOException("Invalid response received from server");
                }
            }
            catch (IOException)
            {
                CloseWithoutNotify();
                throw;
            }
        }

        /// <summary>
        /// Close the connection to the server
        /// </summary>
        public void Close()
        {
            InfoMessage message = new InfoMessage { Code = InfoMessage.Types.StatusCode.ConnEnd, Message = "" };
            // Don't wait very long for send to succeed
            // We are quiting anyway
            try
            {
                _client.SendTimeout = 1000;
                NetworkMessage.WriteToNetwork(message, _client.GetStream());
            }
            catch (Exception)
            {
                // We don't care about quit notification time out or if the server closed anyway
            }
            CloseWithoutNotify();
        }

        /// <summary>
        /// Close the connection without notifing the server.
        /// </summary>
        private void CloseWithoutNotify()
        {
            _client.Close();
        }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    _client.Dispose();
                }


                disposedValue = true;
            }
        }



        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
        }
        #endregion



    }
}
