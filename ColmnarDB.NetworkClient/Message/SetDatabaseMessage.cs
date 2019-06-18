// <auto-generated>
//     Generated by the protocol buffer compiler.  DO NOT EDIT!
//     source: SetDatabaseMessage.proto
// </auto-generated>
#pragma warning disable 1591, 0612, 3021
#region Designer generated code

using pb = global::Google.Protobuf;
using pbc = global::Google.Protobuf.Collections;
using pbr = global::Google.Protobuf.Reflection;
using scg = global::System.Collections.Generic;
namespace ColmnarDB.NetworkClient.Message {

  /// <summary>Holder for reflection information generated from SetDatabaseMessage.proto</summary>
  public static partial class SetDatabaseMessageReflection {

    #region Descriptor
    /// <summary>File descriptor for SetDatabaseMessage.proto</summary>
    public static pbr::FileDescriptor Descriptor {
      get { return descriptor; }
    }
    private static pbr::FileDescriptor descriptor;

    static SetDatabaseMessageReflection() {
      byte[] descriptorData = global::System.Convert.FromBase64String(
          string.Concat(
            "ChhTZXREYXRhYmFzZU1lc3NhZ2UucHJvdG8SH0NvbG1uYXJEQi5OZXR3b3Jr",
            "Q2xpZW50Lk1lc3NhZ2UiKgoSU2V0RGF0YWJhc2VNZXNzYWdlEhQKDERhdGFi",
            "YXNlTmFtZRgBIAEoCWIGcHJvdG8z"));
      descriptor = pbr::FileDescriptor.FromGeneratedCode(descriptorData,
          new pbr::FileDescriptor[] { },
          new pbr::GeneratedClrTypeInfo(null, new pbr::GeneratedClrTypeInfo[] {
            new pbr::GeneratedClrTypeInfo(typeof(global::ColmnarDB.NetworkClient.Message.SetDatabaseMessage), global::ColmnarDB.NetworkClient.Message.SetDatabaseMessage.Parser, new[]{ "DatabaseName" }, null, null, null)
          }));
    }
    #endregion

  }
  #region Messages
  public sealed partial class SetDatabaseMessage : pb::IMessage<SetDatabaseMessage> {
    private static readonly pb::MessageParser<SetDatabaseMessage> _parser = new pb::MessageParser<SetDatabaseMessage>(() => new SetDatabaseMessage());
    private pb::UnknownFieldSet _unknownFields;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pb::MessageParser<SetDatabaseMessage> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::ColmnarDB.NetworkClient.Message.SetDatabaseMessageReflection.Descriptor.MessageTypes[0]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public SetDatabaseMessage() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public SetDatabaseMessage(SetDatabaseMessage other) : this() {
      databaseName_ = other.databaseName_;
      _unknownFields = pb::UnknownFieldSet.Clone(other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public SetDatabaseMessage Clone() {
      return new SetDatabaseMessage(this);
    }

    /// <summary>Field number for the "DatabaseName" field.</summary>
    public const int DatabaseNameFieldNumber = 1;
    private string databaseName_ = "";
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public string DatabaseName {
      get { return databaseName_; }
      set {
        databaseName_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override bool Equals(object other) {
      return Equals(other as SetDatabaseMessage);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public bool Equals(SetDatabaseMessage other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if (DatabaseName != other.DatabaseName) return false;
      return Equals(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override int GetHashCode() {
      int hash = 1;
      if (DatabaseName.Length != 0) hash ^= DatabaseName.GetHashCode();
      if (_unknownFields != null) {
        hash ^= _unknownFields.GetHashCode();
      }
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void WriteTo(pb::CodedOutputStream output) {
      if (DatabaseName.Length != 0) {
        output.WriteRawTag(10);
        output.WriteString(DatabaseName);
      }
      if (_unknownFields != null) {
        _unknownFields.WriteTo(output);
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public int CalculateSize() {
      int size = 0;
      if (DatabaseName.Length != 0) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(DatabaseName);
      }
      if (_unknownFields != null) {
        size += _unknownFields.CalculateSize();
      }
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(SetDatabaseMessage other) {
      if (other == null) {
        return;
      }
      if (other.DatabaseName.Length != 0) {
        DatabaseName = other.DatabaseName;
      }
      _unknownFields = pb::UnknownFieldSet.MergeFrom(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    public void MergeFrom(pb::CodedInputStream input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, input);
            break;
          case 10: {
            DatabaseName = input.ReadString();
            break;
          }
        }
      }
    }

  }

  #endregion

}

#endregion Designer generated code
