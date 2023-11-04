package main 
 
import ( "testing"
         "github.com/stretchr/testify/assert"
)

func TestAntiShuffle(t *testing.T) {
    assert := assert.New(t)
    assert.Equal("Hi", AntiShuffle("Hi"))
    assert.Equal("ehllo", AntiShuffle("hello"))
    assert.Equal("bemnru", AntiShuffle("number"))
    assert.Equal("abcd", AntiShuffle("abcd"))
    assert.Equal("Hello !!!Wdlor", AntiShuffle("Hello World!!!"))
    assert.Equal("", AntiShuffle(""))
    assert.Equal(".Hi My aemn is Meirst .Rboot How aer ?ouy", AntiShuffle("Hi. My name is Mister Robot. How are you?"))
}
