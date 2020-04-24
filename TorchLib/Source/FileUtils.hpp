#include <cstdio>
#include <vector>

#include <sys/stat.h>

namespace FileUtils {

static inline std::uint64_t fileGetLength( FILE * const file )
{
    // https://wiki.sei.cmu.edu/confluence/display/c/FIO19-C.+Do+not+use+fseek%28%29+and+ftell%28%29+to+compute+the+size+of+a+regular+file
    auto const fileDescriptor( ::fileno( file ) );
    struct ::stat info;
    ::fstat( fileDescriptor, &info );
    return info.st_size;
}

static inline std::vector< std::uint8_t > fileReadToBuffer( char const * const filePath )
{
    FILE* file = fopen( filePath, "rb" );
    if ( file == NULL ) return {};
    auto const fileLength( fileGetLength( file ) );

    std::vector< std::uint8_t > fileBuffer;
    fileBuffer.resize( fileLength + 1 );
    fileBuffer[ fileLength ] = '\0';

    std::fread( fileBuffer.data(), 1, fileLength, file );

    fclose(file);

    return fileBuffer;
}

}
