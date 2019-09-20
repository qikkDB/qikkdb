using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace TellStory.Data
{
    public class FileRawData : RawData<FileStream>
    {
        public bool FileExists { get => File.Exists(this.FileName); }

        public string FileName { get; private set; }

        public FileRawData(string fileName) : base()
        {
            this.FileName = fileName;
            Open();
        }

        protected override bool Open()
        {
            if (this.FileName != null && File.Exists(this.FileName) && !this.StreamOpened) this.stream = File.Open(this.FileName, FileMode.Open, FileAccess.Read, FileShare.Read);

            return this.StreamOpened;
        }
    }
}
