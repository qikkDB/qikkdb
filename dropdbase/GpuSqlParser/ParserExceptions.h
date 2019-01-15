//
// Created by Martin Staňo on 2019-01-15.
//

#ifndef DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H
#define DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H

#include <exception>

struct DatabaseNotFoundException : public std::exception {
    const char * what () const noexcept override {
        return "Database was not found";
    }
};

#endif //DROPDBASE_INSTAREA_PARSEREXCEPTIONS_H
