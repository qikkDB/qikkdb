using ColmnarDB.NetworkClient.Message;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Runtime.CompilerServices;
using System.Text;
using Google.Protobuf;
using Google.Protobuf.Collections;
using Google.Protobuf.WellKnownTypes;

namespace ColmnarDB.NetworkClient
{
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

        /// <summary>
        /// Creates new instance of the ColumnarDBClient object
        /// </summary>
        /// <param name="serverHostnameOrIP">IP or hostname of the database server</param>
        /// <param name="serverPort">Port of the database server</param>
        /// <param name="readTimeout">Network read timeout in ms</param>
        /// <param name="writeTimeout">Network read timeout in ms</param>
        public ColumnarDBClient(string serverHostnameOrIP, short serverPort, int readTimeout = 60000, int writeTimeout = 60000)
        {
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

        private (Dictionary<string, List<object>> queryResult, Dictionary<string, float> executionTimes) ConvertToDictionaries(QueryResponseMessage response)
        {
            Dictionary<string, List<object>> queryResult = new Dictionary<string, List<object>>();
            Dictionary<string, float> executionTimes = new Dictionary<string, float>();

            if (response.Timing != null)
            {
                foreach (var record in response.Timing)
                {
                    executionTimes.Add(record.Key, record.Value);
                }
            }

            foreach (var columnData in response.Payloads)
            {
                switch (columnData.Value.PayloadCase)
                {
                    case QueryResponsePayload.PayloadOneofCase.None:
                        throw new ArgumentOutOfRangeException();
                    case QueryResponsePayload.PayloadOneofCase.IntPayload:
                        queryResult.Add(columnData.Key, new List<object>(columnData.Value.IntPayload.IntData.ToArray().Cast<object>()));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.FloatPayload:
                        queryResult.Add(columnData.Key, new List<object>(columnData.Value.FloatPayload.FloatData.ToArray().Cast<object>()));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.Int64Payload:
                        queryResult.Add(columnData.Key, new List<object>(columnData.Value.Int64Payload.Int64Data.ToArray().Cast<object>()));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.DoublePayload:
                        queryResult.Add(columnData.Key, new List<object>(columnData.Value.DoublePayload.DoubleData.ToArray().Cast<object>()));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.PointPayload:
                        queryResult.Add(columnData.Key, new List<object>(columnData.Value.PointPayload.PointData.ToArray().Cast<object>()));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.PolygonPayload:
                        queryResult.Add(columnData.Key, new List<object>(columnData.Value.PolygonPayload.PolygonData.ToArray().Cast<object>()));
                        break;
                    case QueryResponsePayload.PayloadOneofCase.StringPayload:
                        queryResult.Add(columnData.Key, new List<object>(columnData.Value.StringPayload.StringData.ToArray().Cast<object>()));
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
            return (queryResult, executionTimes);
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
        public (Dictionary<string, List<object>> queryResult, Dictionary<string, float> executionTimes) GetNextQueryResult()
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
                            return ( null, null );
                        }
                    }
                    throw new IOException("Invalid response received from server");
                }
                (Dictionary<string, List<object>> queryResult, Dictionary<string, float> executionTimes) resultSet = ConvertToDictionaries(queryResult);
                return resultSet;
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
        public void ImportCSV(string database, string filePath)
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
