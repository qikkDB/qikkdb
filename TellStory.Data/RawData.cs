using System;
using System.IO;

namespace TellStory.Data
{
    public abstract class RawData<T> : GenericData where T:Stream
    {
        protected T stream = null;

        public T Stream { get => this.stream; }

        public bool StreamOpened { get => this.stream != null; }

        public RawData() : base() { }

        public int Write(byte[] buffer, int offset, int count) 
        {
            int retVal = -1;
            if (this.Open() && this.stream.CanWrite)
            {
                this.stream.Write(buffer, offset, count);
                retVal = count;
            }

            return retVal;
        }

        public int Read(byte[] buffer, int offset, int count)
        {
            if (this.Open()) return (this.stream.CanRead ? this.stream.Read(buffer, offset, count) : -1);

            return -1;
        }

        protected virtual bool Open() { return false; }

        protected virtual bool Close()
        {
            if (this.stream == null) return false;

            this.stream.Close();
            this.stream.Dispose();

            this.stream = null;

            return true;
        }

        public void MoveToBegin()
        {
            if (this.stream.CanSeek) this.stream.Seek(0, SeekOrigin.Begin);
        }

        public long Position { get => this.stream.Position; }

        public void MoveToEnd()
        {
            if (this.stream.CanSeek) this.stream.Seek(0, SeekOrigin.End);
        }

        public long Length { get => this.stream.Length; }

        public override void Dispose()
        {
            this.Close();

            base.Dispose();
        }
    }
}
