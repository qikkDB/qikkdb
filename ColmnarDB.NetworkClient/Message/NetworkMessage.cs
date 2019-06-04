using Google.Protobuf;
using Google.Protobuf.WellKnownTypes;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;

namespace ColmnarDB.NetworkClient.Message
{
    public class NetworkMessage
    {
        public static void WriteToNetwork(IMessage message, NetworkStream networkStream)
        {
            var packedMsg = Any.Pack(message);
            int size = packedMsg.CalculateSize();
            byte[] buffer = BitConverter.GetBytes(size);
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(buffer);
            }

            networkStream.Write(buffer, 0, 4);
            packedMsg.WriteTo(networkStream);
        }

        public static Any ReadFromNetwork(NetworkStream networkStream)
        {
            byte[] buffer = new byte[4];
            int totalRead = 0;
            while (totalRead != 4)
            {
                int readNow = networkStream.Read(buffer, totalRead, 4 - totalRead);
                if(readNow == 0)
                {
                    throw new IOException("Conncection was closed by the other side");
                }
                totalRead += readNow;
            }
            totalRead = 0;
            if (BitConverter.IsLittleEndian)
            {
                Array.Reverse(buffer);
            }
            int dataLen = BitConverter.ToInt32(buffer, 0);
            byte[] data = new byte[dataLen];
            while(totalRead != dataLen)
            {
                int readNow = networkStream.Read(data, totalRead, dataLen - totalRead);
                if (readNow == 0)
                {
                    throw new IOException("Conncection was closed by the other side");
                }
                totalRead += readNow;
            }

            return Any.Parser.ParseFrom(data);
        }
    }
}
