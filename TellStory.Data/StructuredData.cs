using System;
using System.Collections.Generic;
using System.Text;

namespace TellStory.Data
{
    public class StructuredData : GenericData
    {
        public string Name { get; protected set; }

        public StructuredData() : base()
        {
            this.Name = Guid.NewGuid().ToString();
        }

        public StructuredData(string name) : base()
        {
            this.Name = name;
        }
    }
}
