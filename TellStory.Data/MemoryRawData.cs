using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace TellStory.Data
{
    public class MemoryRawData : RawData<MemoryStream>
    {
        public MemoryRawData() : base()
        {
            this.stream = new MemoryStream();
        }

        protected override bool Open()
        {
            if (!this.StreamOpened) this.stream = new MemoryStream();

            return this.StreamOpened;
        }
    }
}
