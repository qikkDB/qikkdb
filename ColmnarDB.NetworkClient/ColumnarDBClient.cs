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

        public ColumnarDataTable()
        {
            columnNames = new List<string>();
            columnData = new Dictionary<string, IList>();
            columnTypes = new Dictionary<string, System.Type>();
            size = 0;
        }
        public ColumnarDataTable(List<string> columnNames, Dictionary<string, IList> columnData, Dictionary<string, System.Type> columnTypes)
        {
            this.columnNames = columnNames;
            this.columnData = columnData;
            this.columnTypes = columnTypes;
            size = columnData[columnNames[0]].Count;
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

        /// <summary>
        /// Read timeout for the network stream
        /// </summary>
        private readonly int _readTimeout;

        /// <summary>
        /// Write timeout for the network stream
        /// </summary>
        private readonly int _writeTimeout;

        private const int BULK_IMPORT_FRAGMENT_SIZE = 8192 * 1024;

        /// <summary>
        /// Creates new instance of the ColumnarDBClient object
        /// </summary>
        /// <param name="serverHostnameOrIP">IP or hostname of the database server</param>
        /// <param name="serverPort">Port of the database server</param>
        /// <param name="readTimeout">Network read timeout in ms</param>
        /// <param name="writeTimeout">Network read timeout in ms</param>
        public ColumnarDBClient(string connectionString, int readTimeout = 60000, int writeTimeout = 60000)
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
            _readTimeout = readTimeout;
            _writeTimeout = writeTimeout;
        }


        /// <summary>
        /// Connects to the database server
        /// </summary>
        public void Connect()
        {
            _client = new TcpClient(_serverIP.AddressFamily)
            {
                ReceiveTimeout = _readTimeout,
                SendTimeout = _writeTimeout
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
            ColumnarDataTable ret = new ColumnarDataTable(columnNames, columnDatas, columnTypes);
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

            }
            catch (IOException)
            {
                _client.ReceiveTimeout = _readTimeout;
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
                _client.ReceiveTimeout = _readTimeout * 10;
                var resultMessage = NetworkMessage.ReadFromNetwork(_client.GetStream());
                _client.ReceiveTimeout = _readTimeout;

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
                _client.ReceiveTimeout = _readTimeout;
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
                    int typeSize = 1;
                    byte[] dataBuffer = null;
                    int i = 0;
                    switch (type.Name)
                    {
                        case nameof(Byte):
                            columnType = DataType.ColumnInt8T;
                            dataBuffer = new byte[size];
                            ((List<byte>)(dataTable.GetColumnData()[column])).CopyTo(dataBuffer, 0);
                            break;
                        case nameof(Int32):
                            columnType = DataType.ColumnInt;
                            size *= sizeof(int);
                            typeSize = sizeof(int);
                            dataBuffer = new byte[size];
                            Buffer.BlockCopy(((List<Int32>)(dataTable.GetColumnData()[column])).ToArray(), 0, dataBuffer, 0, size);
                            break;
                        case nameof(Int64):
                            columnType = DataType.ColumnLong;
                            size *= sizeof(long);
                            typeSize = sizeof(long);
                            dataBuffer = new byte[size];
                            Buffer.BlockCopy(((List<Int64>)(dataTable.GetColumnData()[column])).ToArray(), 0, dataBuffer, 0, size);
                            break;
                        case nameof(Single):
                            columnType = DataType.ColumnFloat;
                            size *= sizeof(float);
                            typeSize = sizeof(float);
                            dataBuffer = new byte[size];
                            Buffer.BlockCopy(((List<Single>)(dataTable.GetColumnData()[column])).ToArray(), 0, dataBuffer, 0, size);
                            break;
                        case nameof(Double):
                            columnType = DataType.ColumnDouble;
                            size *= sizeof(double);
                            typeSize = sizeof(double);
                            dataBuffer = new byte[size];
                            Buffer.BlockCopy(((List<Double>)(dataTable.GetColumnData()[column])).ToArray(), 0, dataBuffer, 0, size);
                            break;
                        case nameof(Point):
                            columnType = DataType.ColumnPoint;
                            size = 0;
                            foreach (var elem in ((List<Point>)(dataTable.GetColumnData()[column])))
                            {
                                size += sizeof(int) + elem.CalculateSize();
                            }
                            dataBuffer = new byte[size];
                            i = 0;
                            foreach (var elem in ((List<Point>)(dataTable.GetColumnData()[column])))
                            {
                                int len = elem.CalculateSize();
                                unsafe
                                {
                                    byte* lenBytes = (byte*)&len;
                                    for (int j = 0; j < sizeof(int); j++)
                                    {
                                        dataBuffer[i + j] = *lenBytes;
                                        lenBytes++;
                                    }
                                }
                                i += 4;
                                Buffer.BlockCopy((elem).ToByteArray(), 0, dataBuffer, i, len);
                                i += len;
                            }
                            break;
                        case nameof(ComplexPolygon):
                            columnType = DataType.ColumnPolygon;
                            size = 0;
                            foreach (var elem in ((List<ComplexPolygon>)(dataTable.GetColumnData()[column])))
                            {
                                size += sizeof(int) + elem.CalculateSize();
                            }
                            dataBuffer = new byte[size];
                            i = 0;
                            foreach (var elem in ((List<ComplexPolygon>)(dataTable.GetColumnData()[column])))
                            {
                                int len = elem.CalculateSize();
                                unsafe
                                {
                                    byte* lenBytes = (byte*)&len;
                                    for (int j = 0; j < sizeof(int); j++)
                                    {
                                        dataBuffer[i + j] = *lenBytes;
                                        lenBytes++;
                                    }
                                }
                                i += 4;
                                Buffer.BlockCopy((elem).ToByteArray(), 0, dataBuffer, i, len);
                                i += len;
                            }
                            break;
                        case nameof(String):
                            columnType = DataType.ColumnString;
                            size = 0;
                            foreach (var elem in ((List<String>)(dataTable.GetColumnData()[column])))
                            {
                                size += sizeof(int) + elem.Length;
                            }
                            dataBuffer = new byte[size];
                            i = 0;
                            foreach (var elem in ((List<String>)(dataTable.GetColumnData()[column])))
                            {
                                int len = elem.Length;
                                unsafe
                                {
                                    byte* lenBytes = (byte*)&len;
                                    for (int j = 0; j < sizeof(int); j++)
                                    {
                                        dataBuffer[i + j] = *lenBytes;
                                        lenBytes++;
                                    }
                                }
                                i += 4;
                                Buffer.BlockCopy(Encoding.UTF8.GetBytes(elem), 0, dataBuffer, i, len);
                                i += len;
                            }
                            break;
                    }

                    for (i = 0; i < size; i += BULK_IMPORT_FRAGMENT_SIZE)
                    {
                        int fragmentSize = size - i < BULK_IMPORT_FRAGMENT_SIZE ? size - i : BULK_IMPORT_FRAGMENT_SIZE;
                        byte[] smallBuffer = new byte[fragmentSize];
                        Buffer.BlockCopy(dataBuffer, i, smallBuffer, 0, fragmentSize);
                        BulkImportMessage bulkImportMessage = new BulkImportMessage()
                        { TableName = tableName, ElemCount = fragmentSize / typeSize, ColumnName = column, ColumnType = columnType };
                        NetworkMessage.WriteToNetwork(bulkImportMessage, _client.GetStream());
                        NetworkMessage.WriteRaw(_client.GetStream(), smallBuffer, fragmentSize);
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
