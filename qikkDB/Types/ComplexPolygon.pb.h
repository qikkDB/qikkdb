// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: Types/ComplexPolygon.proto

#ifndef GOOGLE_PROTOBUF_INCLUDED_Types_2fComplexPolygon_2eproto
#define GOOGLE_PROTOBUF_INCLUDED_Types_2fComplexPolygon_2eproto

#include <limits>
#include <string>

#include <google/protobuf/port_def.inc>
#if PROTOBUF_VERSION < 3010000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers. Please update
#error your headers.
#endif
#if 3010000 < PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers. Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/port_undef.inc>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_table_driven.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/inlined_string_field.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>  // IWYU pragma: export
#include <google/protobuf/extension_set.h>  // IWYU pragma: export
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
#define PROTOBUF_INTERNAL_EXPORT_Types_2fComplexPolygon_2eproto
PROTOBUF_NAMESPACE_OPEN
namespace internal {
class AnyMetadata;
}  // namespace internal
PROTOBUF_NAMESPACE_CLOSE

// Internal implementation detail -- do not use these members.
struct TableStruct_Types_2fComplexPolygon_2eproto {
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTableField entries[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::AuxillaryParseTableField aux[]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::ParseTable schema[3]
    PROTOBUF_SECTION_VARIABLE(protodesc_cold);
  static const ::PROTOBUF_NAMESPACE_ID::internal::FieldMetadata field_metadata[];
  static const ::PROTOBUF_NAMESPACE_ID::internal::SerializationTable serialization_table[];
  static const ::PROTOBUF_NAMESPACE_ID::uint32 offsets[];
};
extern const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_Types_2fComplexPolygon_2eproto;
namespace QikkDB {
namespace Types {
class ComplexPolygon;
class ComplexPolygonDefaultTypeInternal;
extern ComplexPolygonDefaultTypeInternal _ComplexPolygon_default_instance_;
class GeoPoint;
class GeoPointDefaultTypeInternal;
extern GeoPointDefaultTypeInternal _GeoPoint_default_instance_;
class Polygon;
class PolygonDefaultTypeInternal;
extern PolygonDefaultTypeInternal _Polygon_default_instance_;
}  // namespace Types
}  // namespace QikkDB
PROTOBUF_NAMESPACE_OPEN
template<> ::QikkDB::Types::ComplexPolygon* Arena::CreateMaybeMessage<::QikkDB::Types::ComplexPolygon>(Arena*);
template<> ::QikkDB::Types::GeoPoint* Arena::CreateMaybeMessage<::QikkDB::Types::GeoPoint>(Arena*);
template<> ::QikkDB::Types::Polygon* Arena::CreateMaybeMessage<::QikkDB::Types::Polygon>(Arena*);
PROTOBUF_NAMESPACE_CLOSE
namespace QikkDB {
namespace Types {

// ===================================================================

class ComplexPolygon :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:QikkDB.Types.ComplexPolygon) */ {
 public:
  ComplexPolygon();
  virtual ~ComplexPolygon();

  ComplexPolygon(const ComplexPolygon& from);
  ComplexPolygon(ComplexPolygon&& from) noexcept
    : ComplexPolygon() {
    *this = ::std::move(from);
  }

  inline ComplexPolygon& operator=(const ComplexPolygon& from) {
    CopyFrom(from);
    return *this;
  }
  inline ComplexPolygon& operator=(ComplexPolygon&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const ComplexPolygon& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const ComplexPolygon* internal_default_instance() {
    return reinterpret_cast<const ComplexPolygon*>(
               &_ComplexPolygon_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    0;

  friend void swap(ComplexPolygon& a, ComplexPolygon& b) {
    a.Swap(&b);
  }
  inline void Swap(ComplexPolygon* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline ComplexPolygon* New() const final {
    return CreateMaybeMessage<ComplexPolygon>(nullptr);
  }

  ComplexPolygon* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<ComplexPolygon>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const ComplexPolygon& from);
  void MergeFrom(const ComplexPolygon& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* InternalSerializeWithCachedSizesToArray(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(ComplexPolygon* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "QikkDB.Types.ComplexPolygon";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_Types_2fComplexPolygon_2eproto);
    return ::descriptor_table_Types_2fComplexPolygon_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kPolygonsFieldNumber = 1,
  };
  // repeated .QikkDB.Types.Polygon polygons = 1;
  int polygons_size() const;
  private:
  int _internal_polygons_size() const;
  public:
  void clear_polygons();
  ::QikkDB::Types::Polygon* mutable_polygons(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::QikkDB::Types::Polygon >*
      mutable_polygons();
  private:
  const ::QikkDB::Types::Polygon& _internal_polygons(int index) const;
  ::QikkDB::Types::Polygon* _internal_add_polygons();
  public:
  const ::QikkDB::Types::Polygon& polygons(int index) const;
  ::QikkDB::Types::Polygon* add_polygons();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::QikkDB::Types::Polygon >&
      polygons() const;

  // @@protoc_insertion_point(class_scope:QikkDB.Types.ComplexPolygon)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::QikkDB::Types::Polygon > polygons_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_Types_2fComplexPolygon_2eproto;
};
// -------------------------------------------------------------------

class Polygon :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:QikkDB.Types.Polygon) */ {
 public:
  Polygon();
  virtual ~Polygon();

  Polygon(const Polygon& from);
  Polygon(Polygon&& from) noexcept
    : Polygon() {
    *this = ::std::move(from);
  }

  inline Polygon& operator=(const Polygon& from) {
    CopyFrom(from);
    return *this;
  }
  inline Polygon& operator=(Polygon&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const Polygon& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const Polygon* internal_default_instance() {
    return reinterpret_cast<const Polygon*>(
               &_Polygon_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    1;

  friend void swap(Polygon& a, Polygon& b) {
    a.Swap(&b);
  }
  inline void Swap(Polygon* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline Polygon* New() const final {
    return CreateMaybeMessage<Polygon>(nullptr);
  }

  Polygon* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<Polygon>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const Polygon& from);
  void MergeFrom(const Polygon& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* InternalSerializeWithCachedSizesToArray(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(Polygon* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "QikkDB.Types.Polygon";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_Types_2fComplexPolygon_2eproto);
    return ::descriptor_table_Types_2fComplexPolygon_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kGeoPointsFieldNumber = 1,
  };
  // repeated .QikkDB.Types.GeoPoint geoPoints = 1;
  int geopoints_size() const;
  private:
  int _internal_geopoints_size() const;
  public:
  void clear_geopoints();
  ::QikkDB::Types::GeoPoint* mutable_geopoints(int index);
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::QikkDB::Types::GeoPoint >*
      mutable_geopoints();
  private:
  const ::QikkDB::Types::GeoPoint& _internal_geopoints(int index) const;
  ::QikkDB::Types::GeoPoint* _internal_add_geopoints();
  public:
  const ::QikkDB::Types::GeoPoint& geopoints(int index) const;
  ::QikkDB::Types::GeoPoint* add_geopoints();
  const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::QikkDB::Types::GeoPoint >&
      geopoints() const;

  // @@protoc_insertion_point(class_scope:QikkDB.Types.Polygon)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::QikkDB::Types::GeoPoint > geopoints_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_Types_2fComplexPolygon_2eproto;
};
// -------------------------------------------------------------------

class GeoPoint :
    public ::PROTOBUF_NAMESPACE_ID::Message /* @@protoc_insertion_point(class_definition:QikkDB.Types.GeoPoint) */ {
 public:
  GeoPoint();
  virtual ~GeoPoint();

  GeoPoint(const GeoPoint& from);
  GeoPoint(GeoPoint&& from) noexcept
    : GeoPoint() {
    *this = ::std::move(from);
  }

  inline GeoPoint& operator=(const GeoPoint& from) {
    CopyFrom(from);
    return *this;
  }
  inline GeoPoint& operator=(GeoPoint&& from) noexcept {
    if (GetArenaNoVirtual() == from.GetArenaNoVirtual()) {
      if (this != &from) InternalSwap(&from);
    } else {
      CopyFrom(from);
    }
    return *this;
  }

  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* descriptor() {
    return GetDescriptor();
  }
  static const ::PROTOBUF_NAMESPACE_ID::Descriptor* GetDescriptor() {
    return GetMetadataStatic().descriptor;
  }
  static const ::PROTOBUF_NAMESPACE_ID::Reflection* GetReflection() {
    return GetMetadataStatic().reflection;
  }
  static const GeoPoint& default_instance();

  static void InitAsDefaultInstance();  // FOR INTERNAL USE ONLY
  static inline const GeoPoint* internal_default_instance() {
    return reinterpret_cast<const GeoPoint*>(
               &_GeoPoint_default_instance_);
  }
  static constexpr int kIndexInFileMessages =
    2;

  friend void swap(GeoPoint& a, GeoPoint& b) {
    a.Swap(&b);
  }
  inline void Swap(GeoPoint* other) {
    if (other == this) return;
    InternalSwap(other);
  }

  // implements Message ----------------------------------------------

  inline GeoPoint* New() const final {
    return CreateMaybeMessage<GeoPoint>(nullptr);
  }

  GeoPoint* New(::PROTOBUF_NAMESPACE_ID::Arena* arena) const final {
    return CreateMaybeMessage<GeoPoint>(arena);
  }
  void CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) final;
  void CopyFrom(const GeoPoint& from);
  void MergeFrom(const GeoPoint& from);
  PROTOBUF_ATTRIBUTE_REINITIALIZES void Clear() final;
  bool IsInitialized() const final;

  size_t ByteSizeLong() const final;
  const char* _InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) final;
  ::PROTOBUF_NAMESPACE_ID::uint8* InternalSerializeWithCachedSizesToArray(
      ::PROTOBUF_NAMESPACE_ID::uint8* target, ::PROTOBUF_NAMESPACE_ID::io::EpsCopyOutputStream* stream) const final;
  int GetCachedSize() const final { return _cached_size_.Get(); }

  private:
  inline void SharedCtor();
  inline void SharedDtor();
  void SetCachedSize(int size) const final;
  void InternalSwap(GeoPoint* other);
  friend class ::PROTOBUF_NAMESPACE_ID::internal::AnyMetadata;
  static ::PROTOBUF_NAMESPACE_ID::StringPiece FullMessageName() {
    return "QikkDB.Types.GeoPoint";
  }
  private:
  inline ::PROTOBUF_NAMESPACE_ID::Arena* GetArenaNoVirtual() const {
    return nullptr;
  }
  inline void* MaybeArenaPtr() const {
    return nullptr;
  }
  public:

  ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadata() const final;
  private:
  static ::PROTOBUF_NAMESPACE_ID::Metadata GetMetadataStatic() {
    ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&::descriptor_table_Types_2fComplexPolygon_2eproto);
    return ::descriptor_table_Types_2fComplexPolygon_2eproto.file_level_metadata[kIndexInFileMessages];
  }

  public:

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  enum : int {
    kLatitudeFieldNumber = 1,
    kLongitudeFieldNumber = 2,
  };
  // float latitude = 1;
  void clear_latitude();
  float latitude() const;
  void set_latitude(float value);
  private:
  float _internal_latitude() const;
  void _internal_set_latitude(float value);
  public:

  // float longitude = 2;
  void clear_longitude();
  float longitude() const;
  void set_longitude(float value);
  private:
  float _internal_longitude() const;
  void _internal_set_longitude(float value);
  public:

  // @@protoc_insertion_point(class_scope:QikkDB.Types.GeoPoint)
 private:
  class _Internal;

  ::PROTOBUF_NAMESPACE_ID::internal::InternalMetadataWithArena _internal_metadata_;
  float latitude_;
  float longitude_;
  mutable ::PROTOBUF_NAMESPACE_ID::internal::CachedSize _cached_size_;
  friend struct ::TableStruct_Types_2fComplexPolygon_2eproto;
};
// ===================================================================


// ===================================================================

#ifdef __GNUC__
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__
// ComplexPolygon

// repeated .QikkDB.Types.Polygon polygons = 1;
inline int ComplexPolygon::_internal_polygons_size() const {
  return polygons_.size();
}
inline int ComplexPolygon::polygons_size() const {
  return _internal_polygons_size();
}
inline void ComplexPolygon::clear_polygons() {
  polygons_.Clear();
}
inline ::QikkDB::Types::Polygon* ComplexPolygon::mutable_polygons(int index) {
  // @@protoc_insertion_point(field_mutable:QikkDB.Types.ComplexPolygon.polygons)
  return polygons_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::QikkDB::Types::Polygon >*
ComplexPolygon::mutable_polygons() {
  // @@protoc_insertion_point(field_mutable_list:QikkDB.Types.ComplexPolygon.polygons)
  return &polygons_;
}
inline const ::QikkDB::Types::Polygon& ComplexPolygon::_internal_polygons(int index) const {
  return polygons_.Get(index);
}
inline const ::QikkDB::Types::Polygon& ComplexPolygon::polygons(int index) const {
  // @@protoc_insertion_point(field_get:QikkDB.Types.ComplexPolygon.polygons)
  return _internal_polygons(index);
}
inline ::QikkDB::Types::Polygon* ComplexPolygon::_internal_add_polygons() {
  return polygons_.Add();
}
inline ::QikkDB::Types::Polygon* ComplexPolygon::add_polygons() {
  // @@protoc_insertion_point(field_add:QikkDB.Types.ComplexPolygon.polygons)
  return _internal_add_polygons();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::QikkDB::Types::Polygon >&
ComplexPolygon::polygons() const {
  // @@protoc_insertion_point(field_list:QikkDB.Types.ComplexPolygon.polygons)
  return polygons_;
}

// -------------------------------------------------------------------

// Polygon

// repeated .QikkDB.Types.GeoPoint geoPoints = 1;
inline int Polygon::_internal_geopoints_size() const {
  return geopoints_.size();
}
inline int Polygon::geopoints_size() const {
  return _internal_geopoints_size();
}
inline void Polygon::clear_geopoints() {
  geopoints_.Clear();
}
inline ::QikkDB::Types::GeoPoint* Polygon::mutable_geopoints(int index) {
  // @@protoc_insertion_point(field_mutable:QikkDB.Types.Polygon.geoPoints)
  return geopoints_.Mutable(index);
}
inline ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::QikkDB::Types::GeoPoint >*
Polygon::mutable_geopoints() {
  // @@protoc_insertion_point(field_mutable_list:QikkDB.Types.Polygon.geoPoints)
  return &geopoints_;
}
inline const ::QikkDB::Types::GeoPoint& Polygon::_internal_geopoints(int index) const {
  return geopoints_.Get(index);
}
inline const ::QikkDB::Types::GeoPoint& Polygon::geopoints(int index) const {
  // @@protoc_insertion_point(field_get:QikkDB.Types.Polygon.geoPoints)
  return _internal_geopoints(index);
}
inline ::QikkDB::Types::GeoPoint* Polygon::_internal_add_geopoints() {
  return geopoints_.Add();
}
inline ::QikkDB::Types::GeoPoint* Polygon::add_geopoints() {
  // @@protoc_insertion_point(field_add:QikkDB.Types.Polygon.geoPoints)
  return _internal_add_geopoints();
}
inline const ::PROTOBUF_NAMESPACE_ID::RepeatedPtrField< ::QikkDB::Types::GeoPoint >&
Polygon::geopoints() const {
  // @@protoc_insertion_point(field_list:QikkDB.Types.Polygon.geoPoints)
  return geopoints_;
}

// -------------------------------------------------------------------

// GeoPoint

// float latitude = 1;
inline void GeoPoint::clear_latitude() {
  latitude_ = 0;
}
inline float GeoPoint::_internal_latitude() const {
  return latitude_;
}
inline float GeoPoint::latitude() const {
  // @@protoc_insertion_point(field_get:QikkDB.Types.GeoPoint.latitude)
  return _internal_latitude();
}
inline void GeoPoint::_internal_set_latitude(float value) {
  
  latitude_ = value;
}
inline void GeoPoint::set_latitude(float value) {
  _internal_set_latitude(value);
  // @@protoc_insertion_point(field_set:QikkDB.Types.GeoPoint.latitude)
}

// float longitude = 2;
inline void GeoPoint::clear_longitude() {
  longitude_ = 0;
}
inline float GeoPoint::_internal_longitude() const {
  return longitude_;
}
inline float GeoPoint::longitude() const {
  // @@protoc_insertion_point(field_get:QikkDB.Types.GeoPoint.longitude)
  return _internal_longitude();
}
inline void GeoPoint::_internal_set_longitude(float value) {
  
  longitude_ = value;
}
inline void GeoPoint::set_longitude(float value) {
  _internal_set_longitude(value);
  // @@protoc_insertion_point(field_set:QikkDB.Types.GeoPoint.longitude)
}

#ifdef __GNUC__
  #pragma GCC diagnostic pop
#endif  // __GNUC__
// -------------------------------------------------------------------

// -------------------------------------------------------------------


// @@protoc_insertion_point(namespace_scope)

}  // namespace Types
}  // namespace QikkDB

// @@protoc_insertion_point(global_scope)

#include <google/protobuf/port_undef.inc>
#endif  // GOOGLE_PROTOBUF_INCLUDED_GOOGLE_PROTOBUF_INCLUDED_Types_2fComplexPolygon_2eproto
