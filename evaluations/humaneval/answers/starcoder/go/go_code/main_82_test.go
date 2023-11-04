package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestPrimeLength(t *testing.T) {
    assert := assert.New(t)
    assert.Equal(true, PrimeLength("Hello"))
    assert.Equal(true, PrimeLength("abcdcba"))
    assert.Equal(true, PrimeLength("kittens"))
    assert.Equal(false, PrimeLength("orange"))
    assert.Equal(true, PrimeLength("wow"))
    assert.Equal(true, PrimeLength("world"))
    assert.Equal(true, PrimeLength("MadaM"))
    assert.Equal(true, PrimeLength("Wow"))
    assert.Equal(false, PrimeLength(""))
    assert.Equal(true, PrimeLength("HI"))
    assert.Equal(true, PrimeLength("go"))
    assert.Equal(false, PrimeLength("gogo"))
    assert.Equal(false, PrimeLength("aaaaaaaaaaaaaaa"))
    assert.Equal(true, PrimeLength("Madam"))
    assert.Equal(false, PrimeLength("M"))
    assert.Equal(false, PrimeLength("0"))
}
