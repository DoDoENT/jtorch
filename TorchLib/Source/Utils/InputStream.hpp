#pragma once

#include <cstdint>
#include <vector>

namespace mtorch
{

template< typename T >
union Aliased
{
    using decayed_type = std::decay_t< T >;

    decayed_type  value_;
    std::uint8_t  bytes_[ sizeof( decayed_type ) ];

    constexpr auto operator[]( std::size_t index ) const noexcept
    {
        return bytes_[ index ];
    }

    constexpr auto & operator[]( std::size_t index ) noexcept
    {
        return bytes_[ index ];
    }

    constexpr auto size()  const noexcept { return sizeof( decayed_type ); }

    constexpr auto begin() const noexcept { return bytes_; }
    constexpr auto begin()       noexcept { return bytes_; }

    constexpr auto end()   const noexcept { return bytes_ + size(); }
    constexpr auto end()         noexcept { return bytes_ + size(); }
};

class InputStream
{
public:
    InputStream( std::vector< uint8_t > const & buffer ) noexcept :
        buffer_( buffer ),
        currentPos_( buffer.data() )
    {}

    template< typename T >
    T read() noexcept
    {
        Aliased< T > a;
        std::memcpy( a.begin(), currentPos_, a.size() );
        currentPos_ += a.size();
        return a.value_;
    }

    template< typename T >
    void readArray( T * dest, std::size_t numElements )
    {
        auto numBytes = numElements * sizeof( T );
        std::memcpy( dest, currentPos_, numBytes );
        currentPos_ += numBytes;
    }

private:
    std::vector< std::uint8_t > const   buffer_;
    std::uint8_t                const * currentPos_;
};

}
