// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: messages/InfoMessage.proto

#include "messages/InfoMessage.pb.h"

#include <algorithm>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
namespace QikkDB {
namespace NetworkClient {
namespace Message {
class InfoMessageDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<InfoMessage> _instance;
} _InfoMessage_default_instance_;
}  // namespace Message
}  // namespace NetworkClient
}  // namespace QikkDB
static void InitDefaultsscc_info_InfoMessage_messages_2fInfoMessage_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::QikkDB::NetworkClient::Message::_InfoMessage_default_instance_;
    new (ptr) ::QikkDB::NetworkClient::Message::InfoMessage();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::QikkDB::NetworkClient::Message::InfoMessage::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_InfoMessage_messages_2fInfoMessage_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, 0, InitDefaultsscc_info_InfoMessage_messages_2fInfoMessage_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_messages_2fInfoMessage_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_messages_2fInfoMessage_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_messages_2fInfoMessage_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_messages_2fInfoMessage_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::QikkDB::NetworkClient::Message::InfoMessage, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::QikkDB::NetworkClient::Message::InfoMessage, code_),
  PROTOBUF_FIELD_OFFSET(::QikkDB::NetworkClient::Message::InfoMessage, message_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::QikkDB::NetworkClient::Message::InfoMessage)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::QikkDB::NetworkClient::Message::_InfoMessage_default_instance_),
};

const char descriptor_table_protodef_messages_2fInfoMessage_2eproto[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) =
  "\n\032messages/InfoMessage.proto\022\034QikkDB.Net"
  "workClient.Message\"\354\001\n\013InfoMessage\022B\n\004Co"
  "de\030\001 \001(\01624.QikkDB.NetworkClient.Message."
  "InfoMessage.StatusCode\022\017\n\007Message\030\002 \001(\t\""
  "\207\001\n\nStatusCode\022\006\n\002OK\020\000\022\010\n\004WAIT\020\001\022\023\n\017GET_"
  "NEXT_RESULT\020\006\022\017\n\013QUERY_ERROR\020\002\022\020\n\014IMPORT"
  "_ERROR\020\003\022\022\n\016CONN_ESTABLISH\020\004\022\014\n\010CONN_END"
  "\020\005\022\r\n\tHEARTBEAT\020\007b\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_messages_2fInfoMessage_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_messages_2fInfoMessage_2eproto_sccs[1] = {
  &scc_info_InfoMessage_messages_2fInfoMessage_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_messages_2fInfoMessage_2eproto_once;
static bool descriptor_table_messages_2fInfoMessage_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_messages_2fInfoMessage_2eproto = {
  &descriptor_table_messages_2fInfoMessage_2eproto_initialized, descriptor_table_protodef_messages_2fInfoMessage_2eproto, "messages/InfoMessage.proto", 305,
  &descriptor_table_messages_2fInfoMessage_2eproto_once, descriptor_table_messages_2fInfoMessage_2eproto_sccs, descriptor_table_messages_2fInfoMessage_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_messages_2fInfoMessage_2eproto::offsets,
  file_level_metadata_messages_2fInfoMessage_2eproto, 1, file_level_enum_descriptors_messages_2fInfoMessage_2eproto, file_level_service_descriptors_messages_2fInfoMessage_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_messages_2fInfoMessage_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_messages_2fInfoMessage_2eproto), true);
namespace QikkDB {
namespace NetworkClient {
namespace Message {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* InfoMessage_StatusCode_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_messages_2fInfoMessage_2eproto);
  return file_level_enum_descriptors_messages_2fInfoMessage_2eproto[0];
}
bool InfoMessage_StatusCode_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
    case 5:
    case 6:
    case 7:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)
constexpr InfoMessage_StatusCode InfoMessage::OK;
constexpr InfoMessage_StatusCode InfoMessage::WAIT;
constexpr InfoMessage_StatusCode InfoMessage::GET_NEXT_RESULT;
constexpr InfoMessage_StatusCode InfoMessage::QUERY_ERROR;
constexpr InfoMessage_StatusCode InfoMessage::IMPORT_ERROR;
constexpr InfoMessage_StatusCode InfoMessage::CONN_ESTABLISH;
constexpr InfoMessage_StatusCode InfoMessage::CONN_END;
constexpr InfoMessage_StatusCode InfoMessage::HEARTBEAT;
constexpr InfoMessage_StatusCode InfoMessage::StatusCode_MIN;
constexpr InfoMessage_StatusCode InfoMessage::StatusCode_MAX;
constexpr int InfoMessage::StatusCode_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)

// ===================================================================

void InfoMessage::InitAsDefaultInstance() {
}
class InfoMessage::_Internal {
 public:
};

InfoMessage::InfoMessage()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:QikkDB.NetworkClient.Message.InfoMessage)
}
InfoMessage::InfoMessage(const InfoMessage& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  message_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (!from._internal_message().empty()) {
    message_.AssignWithDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.message_);
  }
  code_ = from.code_;
  // @@protoc_insertion_point(copy_constructor:QikkDB.NetworkClient.Message.InfoMessage)
}

void InfoMessage::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_InfoMessage_messages_2fInfoMessage_2eproto.base);
  message_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  code_ = 0;
}

InfoMessage::~InfoMessage() {
  // @@protoc_insertion_point(destructor:QikkDB.NetworkClient.Message.InfoMessage)
  SharedDtor();
}

void InfoMessage::SharedDtor() {
  message_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void InfoMessage::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const InfoMessage& InfoMessage::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_InfoMessage_messages_2fInfoMessage_2eproto.base);
  return *internal_default_instance();
}


void InfoMessage::Clear() {
// @@protoc_insertion_point(message_clear_start:QikkDB.NetworkClient.Message.InfoMessage)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  message_.ClearToEmptyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  code_ = 0;
  _internal_metadata_.Clear();
}

const char* InfoMessage::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // .QikkDB.NetworkClient.Message.InfoMessage.StatusCode Code = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
          _internal_set_code(static_cast<::QikkDB::NetworkClient::Message::InfoMessage_StatusCode>(val));
        } else goto handle_unusual;
        continue;
      // string Message = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParserUTF8(_internal_mutable_message(), ptr, ctx, "QikkDB.NetworkClient.Message.InfoMessage.Message");
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}

::PROTOBUF_NAMESPACE_ID::uint8* InfoMessage::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const {
  // @@protoc_insertion_point(serialize_to_array_start:QikkDB.NetworkClient.Message.InfoMessage)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // .QikkDB.NetworkClient.Message.InfoMessage.StatusCode Code = 1;
  if (this->code() != 0) {
    stream->EnsureSpace(&target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      1, this->_internal_code(), target);
  }

  // string Message = 2;
  if (this->message().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->_internal_message().data(), static_cast<int>(this->_internal_message().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "QikkDB.NetworkClient.Message.InfoMessage.Message");
    target = stream->WriteStringMaybeAliased(
        2, this->_internal_message(), target);
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::InternalSerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target, stream);
  }
  // @@protoc_insertion_point(serialize_to_array_end:QikkDB.NetworkClient.Message.InfoMessage)
  return target;
}

size_t InfoMessage::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:QikkDB.NetworkClient.Message.InfoMessage)
  size_t total_size = 0;

  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string Message = 2;
  if (this->message().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->_internal_message());
  }

  // .QikkDB.NetworkClient.Message.InfoMessage.StatusCode Code = 1;
  if (this->code() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->_internal_code());
  }

  if (PROTOBUF_PREDICT_FALSE(_internal_metadata_.have_unknown_fields())) {
    return ::PROTOBUF_NAMESPACE_ID::internal::ComputeUnknownFieldsSize(
        _internal_metadata_, total_size, &_cached_size_);
  }
  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void InfoMessage::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:QikkDB.NetworkClient.Message.InfoMessage)
  GOOGLE_DCHECK_NE(&from, this);
  const InfoMessage* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<InfoMessage>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:QikkDB.NetworkClient.Message.InfoMessage)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:QikkDB.NetworkClient.Message.InfoMessage)
    MergeFrom(*source);
  }
}

void InfoMessage::MergeFrom(const InfoMessage& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:QikkDB.NetworkClient.Message.InfoMessage)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.message().size() > 0) {

    message_.AssignWithDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.message_);
  }
  if (from.code() != 0) {
    _internal_set_code(from._internal_code());
  }
}

void InfoMessage::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:QikkDB.NetworkClient.Message.InfoMessage)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void InfoMessage::CopyFrom(const InfoMessage& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:QikkDB.NetworkClient.Message.InfoMessage)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool InfoMessage::IsInitialized() const {
  return true;
}

void InfoMessage::InternalSwap(InfoMessage* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  message_.Swap(&other->message_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(code_, other->code_);
}

::PROTOBUF_NAMESPACE_ID::Metadata InfoMessage::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace Message
}  // namespace NetworkClient
}  // namespace QikkDB
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::QikkDB::NetworkClient::Message::InfoMessage* Arena::CreateMaybeMessage< ::QikkDB::NetworkClient::Message::InfoMessage >(Arena* arena) {
  return Arena::CreateInternal< ::QikkDB::NetworkClient::Message::InfoMessage >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>